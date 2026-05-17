from __future__ import annotations

from tqdm import tqdm

from ..utils.checkpoint import (
    append_inference_checkpoint,
    load_inference_checkpoint,
    restore_llamacpp_results,
    restore_ollama_results,
)
from ..utils.utils import *
from .format import Response, create_format, schema_json
from .Inference import Inference


def _lazy_ollama():
    """Import ollama lazily so the module can be imported without it installed."""
    try:
        import ollama
        from ollama import Client
        return ollama, Client
    except ImportError as exc:
        raise ImportError(
            "InferenceOllama requires the 'ollama' extra. "
            "Install with: pip install 'urban-worm[ollama]'"
        ) from exc


class InferenceOllama(Inference):
    '''
    Constructor for vision inference using MLLMs with Ollama.

    Args:
        llm (str): model checkpoint.
        ollama_key (str): The Ollama API key.
        model_dir (str, optional): Directory where Ollama stores downloaded
            models.  Sets the ``OLLAMA_MODELS`` environment variable before
            each ``ollama.pull`` call.  **Note:** for the running Ollama
            server to store *new* downloads here, it must also have been
            started with ``OLLAMA_MODELS`` pointing to the same directory
            (e.g. ``OLLAMA_MODELS=/data/models ollama serve``).  If the
            server is already running with a different directory, this setting
            affects only where the client *looks*, not where the server saves.
        **kwargs: image (str|list[str]|tuple[str]), images (list|tuple), data constructor (GeoTaggedData), and schema (dict)
    '''

    def __init__(self,
                 llm: str = None,
                 ollama_key: str = None,
                 model_dir: str | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.skip_errors = True
        self.ollama_key = ollama_key
        self.model_dir = model_dir

    def one_inference(self,
                      system: str = '',
                      prompt: str = '',
                      image: str | list[str] | tuple[str] = None,
                      audio: str | list[str] | tuple[str] = None,
                      temp: float = 0.0,
                      top_k: int = 20.0,
                      top_p: float = 0.8,
                      max_new_tokens: int = 512):

        '''
        Chat with MLLM model with one image.

        Args:
            system (str, optional): The system message.
            prompt (str): The prompt message.
            image (str | list[str] | tuple[str]): The image path.
            audio (str | list[str] | tuple[str]): The audio path.
            temp (float): The temperature value.
            top_k (int): The top_k value.
            top_p (float): The top_p value.
            max_new_tokens (int): Maximum number of tokens to generate.  Default 512.

        Notes:
            Ollama currently does not support audio input.
            The argument `audio` is just a placeholder for the future development.

        Returns:
            dict: A dictionary includes questions/messages, responses/answers
        '''

        ollama, _ = _lazy_ollama()
        if self.model_dir is not None:
            import os
            _prev_ollama_models = os.environ.get("OLLAMA_MODELS")
            os.environ["OLLAMA_MODELS"] = self.model_dir
            try:
                ollama.pull(self.llm, stream=True)
            finally:
                if _prev_ollama_models is None:
                    os.environ.pop("OLLAMA_MODELS", None)
                else:
                    os.environ["OLLAMA_MODELS"] = _prev_ollama_models
        else:
            ollama.pull(self.llm, stream=True)
        multiImg = False
        if image is None and audio is not None:
            # Audio is not supported by Ollama yet; fall through with the
            # path treated as an image so the user gets a clear error from
            # the model rather than a NameError here.
            image = audio
        if image is not None:
            img = image
        else:
            img = self.img
        if isinstance(img, list) or isinstance(img, tuple):
            if not isinstance(img[0], str):
                self.logger.warning("a list of images can only be a flatten list")
            multiImg = True
        else:
            img = [img]

        schema = create_format(self.schema)

        dic = {'responses': [], 'data': []}
        r = self._mtmd(model=self.llm,
                       system=system, prompt=prompt,
                       img=img,
                       temp=temp, top_k=top_k, top_p=top_p,
                       num_predict=max_new_tokens,
                       schema=schema,
                       one_shot_lr=[],
                       multiImgInput=multiImg)
        dic['responses'] += [r.responses]
        dic['data'] += [img]
        return response2df(dic)

    def batch_inference(self,
                        system: str = '',
                        prompt: str = '',
                        temp: float = 0.0,
                        top_k: int = 20,
                        top_p: float = 0.8,
                        max_new_tokens: int = 512,
                        disableProgressBar: bool = False,
                        checkpoint_path: str | None = None) -> dict:
        '''
        Chat with MLLM model for each image.

        Args:
            system (str, optional): The system message.
            prompt (str): The prompt message.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
            max_new_tokens (int): Maximum number of tokens to generate per call.  Default 512.
            disableProgressBar (bool): The progress bar for showing the progress of data analysis over the units.
            checkpoint_path (str, optional): Path to a JSONL file for resume-safe checkpointing.
                Already-completed items are skipped on the next run automatically.

        Returns:
            list A list of dictionaries. Each dict includes questions/messages, responses/answers, and image base64 (if required)
        '''

        ollama, _ = _lazy_ollama()
        if self.model_dir is not None:
            import os
            _prev_ollama_models = os.environ.get("OLLAMA_MODELS")
            os.environ["OLLAMA_MODELS"] = self.model_dir
            try:
                ollama.pull(self.llm, stream=True)
            finally:
                if _prev_ollama_models is None:
                    os.environ.pop("OLLAMA_MODELS", None)
                else:
                    os.environ["OLLAMA_MODELS"] = _prev_ollama_models
        else:
            ollama.pull(self.llm, stream=True)

        if self.batch_images is not None:
            imgs = self.batch_images
        else:
            imgs = self.imgs

        schema = create_format(self.schema)

        multiImgInput = False
        if isinstance(imgs[0], list) or isinstance(imgs[0], tuple):
            multiImgInput = True

        # ── resume from checkpoint ───────────────────────────────────────
        done_records = load_inference_checkpoint(checkpoint_path) if checkpoint_path else []
        start_idx = len(done_records)
        dic = restore_ollama_results(done_records)

        for i in tqdm(range(start_idx, len(imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            img = imgs[i]
            try:
                r = self._mtmd(model=self.llm,
                               system=system, prompt=prompt,
                               img=img if multiImgInput else [img],
                               temp=temp, top_k=top_k, top_p=top_p,
                               num_predict=max_new_tokens,
                               schema=schema,
                               one_shot_lr=[],
                               multiImgInput=multiImgInput)
                rr = r.responses
            except Exception as e:
                # Log and continue; capture an error stub so downstream stays consistent
                self.logger.warning("batch_inference: image %d failed (%s). Continuing.", i, e)
                rr = []

            dic['responses'] += [rr]
            dic['data'] += [imgs[i]]

            if checkpoint_path:
                try:
                    responses_dump = [item.model_dump() for item in rr]
                except Exception:
                    responses_dump = [dict(item) for item in rr] if rr else []
                append_inference_checkpoint(checkpoint_path, {
                    'idx': i,
                    'responses': responses_dump,
                    'data': imgs[i] if isinstance(imgs[i], str) else list(imgs[i]),
                })

        self.results = dic
        return self.to_df(output=True)

    def to_df(self, output: bool = True) -> pd.DataFrame | str:
        """
        Convert the output from an MLLM reponse (from .batch_inference) into a DataFrame.

        Args:
            output (bool): Whether to return a DataFrame. Defaults to True.
        Returns:
            pd.DataFrame: A DataFrame containing responses and associated metadata.
            str: An error message if `.batch_inference()` has not been run or if the format is unsupported.
        """

        if self.results is not None:
            self.df = response2df(self.results)
            if output:
                return self.df
        return None

    def _mtmd(self, model: str = None, system: str = None, prompt: str = None,
              img: list[str] = None, temp: float = None, top_k: float = None, top_p: float = None,
              num_predict: int = 512,
              schema = None,
              one_shot_lr: list | tuple | None = None, multiImgInput: bool = False, audio_input: bool = False):
        if one_shot_lr is None:
            one_shot_lr = []

        if prompt is not None and img is not None:
            if len(img) == 1:
                return self._customized_chat(model, system, prompt, img[0], temp, top_k, top_p, num_predict, schema, one_shot_lr)
            elif len(img) >= 2:
                system = (
                    "You are analyzing aerial or street view images. For street "
                    "view, you should just focus on the building and yard in "
                    f"the middle. {system}"
                )
                if multiImgInput:
                    return self._customized_chat(model, system, prompt, img, temp, top_k, top_p, num_predict, schema, one_shot_lr)
                # Per-image inference: pass img[i], not the full list.
                res = []
                for i in range(len(img)):
                    r = self._customized_chat(model, system, prompt, img[i], temp, top_k, top_p, num_predict, schema, one_shot_lr)
                    res += [r.responses]
                return res
            return None
        else:
            raise ValueError("Prompt or image(s) is missing.")

    def _customized_chat(self, model: str = None,
                         system: str = None, prompt: str = None, img: str | list | tuple = None,
                         temp: float = None, top_k: float = None, top_p: float = None,
                         num_predict: int = 512,
                         schema=None,
                         one_shot_lr: list | None = None,
                         audio_input: bool = False) -> Response:
        if one_shot_lr is None:
            one_shot_lr = []
        if isinstance(one_shot_lr, list):
            if len(one_shot_lr) > 0:
                if not isinstance(one_shot_lr[0], dict):
                    raise TypeError("Please provide a list of dictionaries.")

        if img is not None:
            if isinstance(img, str):
                messages = [
                               {
                                   'role': 'system',
                                   'content': system
                               }] + one_shot_lr + [
                               {
                                   'role': 'user',
                                   'content': prompt,
                                   'images': [img]
                               }
                           ]
            elif isinstance(img, list) or isinstance(img, tuple):
                th = ['st', 'nd', 'rd', 'th']
                img_messages = [{'role': 'system', 'content': system}] + one_shot_lr + [
                    {'role': 'user', 'content': f'{i + 1}{th[i] if i < 3 else th[3]} image', 'images': [img[i]]} for i
                    in range(len(img))]
                messages = img_messages + [
                    {
                        'role': 'user',
                        'content': 'You have to answer all questions based on all given images\n' + prompt,
                    }
                ]
        else:
            messages = [
                           {
                               'role': 'system',
                               'content': system
                           }] + one_shot_lr + [
                           {
                               'role': 'user',
                               'content': prompt,
                           }
                       ]

        ollama, Client = _lazy_ollama()
        if (self.ollama_key is not None) and (self.ollama_key != ''):
            client = Client(
                host="https://ollama.com",
                headers={'Authorization': 'Bearer ' + self.ollama_key},
            )
            res = client.chat(
                model=model,
                format=schema.model_json_schema(),
                messages=messages,
                options={
                    "temperature": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "num_predict": num_predict,
                }
            )
        else:
            res = ollama.chat(
                model=model,
                format=schema.model_json_schema(),
                messages=messages,
                options={
                    "temperature": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "num_predict": num_predict,
                }
            )

        raw_text = res.message.content
        try:
            return schema.model_validate_json(raw_text)
        except Exception as direct_err:
            self.logger.debug("direct JSON validation failed: %s", direct_err)

        repaired = sanitize_json_text(str(raw_text))
        try:
            return schema.model_validate_json(repaired)
        except Exception as repair_err:
            self.logger.debug("validation after sanitize failed: %s", repair_err)

        extracted = extract_json_from_text(repaired) or repaired
        try:
            return schema.model_validate_json(extracted)
        except Exception:
            # All recovery paths exhausted. If skip_errors is True, suppress
            # and return an empty Response shape so batch loops can continue.
            if self.skip_errors:
                self.logger.warning(
                    "Could not validate model output against schema; "
                    "returning empty response."
                )
                return schema(responses=[])
            raise


import subprocess
from pathlib import Path

import pandas as pd

from ..utils.utils import extract_last_json, responses_to_wide_all_columns


class InferenceLlamacpp(Inference):
    '''
    Constructor for vision inference using MLLMs with llama.cpp

    Args:
        llm (str, optional): model checkpoint to download (e.g.
            ``ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0``) or a local path
            to a ``.gguf`` model file.
        mp (str, optional): If ``llm`` is a local ``.gguf`` path, ``mp``
            must be the local path to the multimodal projector file
            (``*mmproj*.gguf``).
        model_dir (str, optional): Directory used as ``HF_HUB_CACHE`` when
            llama-mtmd-cli downloads a model via the ``-hf`` flag.  GGUF
            files from HuggingFace will be cached here instead of the
            default ``~/.cache/huggingface/hub``.  Has no effect when
            ``llm`` is already a local file path.
        **kwargs: image (str|list[str]|tuple[str]), images (list|tuple), data constructor (GeoTaggedData), and schema (dict)
    '''

    def __init__(self, llm: str = None, mp: str = None,
                 model_dir: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.mp = mp
        self.model_dir = model_dir

    def one_inference(self,
                      system: str = '',
                      prompt: str = '',
                      image: str | list | tuple = None,
                      audio: str | list | tuple = None,
                      temp: float = 0.2,
                      top_k: int = 20,
                      top_p: float = 0.8,
                      ctx_size: int = 4096,
                      max_new_tokens: int = 512,
                      audio_input: bool = False
                      ) -> Any:
        '''
            Chat with MLLM model with one image.
            Args:
                 system (str, optional): The system message.
                 prompt (str): The prompt message.
                 image (str | list | tuple, optional): The image path.
                 audio (str | list | tuple, optional): The audio path.
                 temp (float): The temperature value.
                 top_k (int): The top_k value.
                 top_p (float): The top_p value.
                 ctx_size (int): Size of context (The default is 4096)
                 max_new_tokens (int): Maximum number of tokens to generate.  Default 512.
                 audio_input (bool, optional): Whether to run inference with audio input

            Returns: response from MLLM as a dataframe
        '''

        llm = self.llm
        mp = self.mp

        if not audio_input:
            if image is not None:
                im = [image] if isinstance(image, str) else image
            else:
                im = [self.img] if isinstance(self.img, str) else self.img

        else:
            if audio is not None:
                im = [audio] if isinstance(audio, str) else audio
            else:
                im = [self.audio] if isinstance(self.audio, str) else self.audio

        if isinstance(im, list) or isinstance(im, tuple):
            if not isinstance(im[0], str):
                self.logger.warning("a list of images can only be a flatten list")
                return None

        # ims_origin = None
        im_ = []
        if not audio_input:
            for i in im:
                if is_base64(i):
                    tmp_path = base64img2temp(i)
                    im_ += [tmp_path]
                elif is_url(i):
                    tmp_path = url2temp(i)
                    im_ += [tmp_path]
                else:
                    pass
        else:
            for i in range(len(im)):
                if is_url(im[i]):
                    tmp_path = sound_url_to_temp(im[i])
                    im_ += [tmp_path]
                else:
                    pass

        if len(im_) == len(im):
            # ims_origin = im
            im = im_

        if llm is None:
            self.logger.warning("model cannot be None")
            return None

        schema = create_format(self.schema)

        r = self._mtmd(llm, mp,
                       system,
                       prompt,
                       im,
                       temperature=temp,
                       top_k=top_k,
                       top_p=top_p,
                       ctx_size=ctx_size,
                       max_new_tokens=max_new_tokens,
                       schema=schema,
                       audio_input=audio_input)
        r = extract_last_json(r)
        r = pd.DataFrame(r['responses'])
        df = responses_to_wide_all_columns(r)
        # df['data'] = ''
        # df.loc[0, 'data'] = im
        if len(im_) >= 1:
            for each in im_:
                try:
                    os.remove(each)
                except OSError:
                    pass
        return df

    def batch_inference(self,
                        system: str = '',
                        prompt: str = '',
                        temp: float = 0.2,
                        top_k: int = 20,
                        top_p: float = 0.8,
                        min_p: float = 0.0,
                        seed: int = 3407,
                        ctx_size: int = 4096,
                        max_new_tokens: int = 512,
                        audio_input = False,
                        disableProgressBar: bool = False,
                        checkpoint_path: str | None = None):
        '''
            Chat with MLLM model for each image in a list.
            Args:
                system (str, optional): The system message.
                prompt (str): The prompt message.
                temp (float): The temperature value (default: 0.2)
                top_k (float): The top_k value (default: 20)
                top_p (float): The top_p value (default: 0.8)
                min_p (float): min-p sampling (default: 0.0, 0.0 = disabled)
                seed (int): The seed value (Default is 3407)
                ctx_size (int): Size of context (Default is 4096)
                max_new_tokens (int): Maximum number of tokens to generate per call.  Default 512.
                audio_input (bool): Whether to run inference with audio input
                disableProgressBar (bool): Whether to disable progress bar.
                checkpoint_path (str, optional): Path to a JSONL file for resume-safe checkpointing.
                    Already-completed items are skipped on the next run automatically.
            Returns: response from MLLM as a dataframe
        '''

        llm = self.llm
        mp = self.mp
        clips = None
        if not audio_input:
            if self.batch_images is not None:
                imgs = self.batch_images
            else:
                imgs = self.imgs
        else:
            if self.batch_audios is not None:
                imgs = self.batch_audios
                clips = self.batch_audios_slice
            else:
                imgs = self.audios

        schema = create_format(self.schema)

        # ── resume from checkpoint ───────────────────────────────────────
        done_records = load_inference_checkpoint(checkpoint_path) if checkpoint_path else []
        start_idx = len(done_records)
        dic = restore_llamacpp_results(done_records)

        for i in tqdm(range(start_idx, len(imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            ims = [imgs[i]] if isinstance(imgs[i], str) else imgs[i]

            ims_origin = None
            ims_ = []
            if not audio_input:
                for im in ims:
                    if is_base64(im):
                        tmp_path = base64img2temp(im)
                        ims_ += [tmp_path]
                    elif is_url(im):
                        tmp_path = url2temp(im)
                        ims_ += [tmp_path]
                    else:
                        pass
            else:
                for j in range(len(ims)):
                    im = ims[j]
                    if is_url(im):
                        if clips is not None:
                            clip_range = clips[j]
                            tmp_path = sound_url_to_temp(im, clip_range)
                            ims_ += [tmp_path]
                        else:
                            tmp_path = sound_url_to_temp(im)
                            ims_ += [tmp_path]
                    else:
                        pass

            if len(ims_) == len(ims):
                ims_origin = ims
                ims = ims_

            try:
                r = None
                try_times = 0
                while r is None and try_times <= 5:
                    r = self._mtmd(llm,
                                   mp,
                                   system,
                                   prompt,
                                   ims,
                                   temperature=temp,
                                   top_k=top_k,
                                   top_p=top_p,
                                   min_p=min_p,
                                   seed=seed,
                                   ctx_size=ctx_size,
                                   max_new_tokens=max_new_tokens,
                                   schema=schema,
                                   audio_input=audio_input)
                    r = extract_last_json(r)
                    try_times += 1

                if r is None:
                    r = 'Bad response'
                dic['responses'] += [r]
                stored_data = ims if ims_origin is None else ims_origin
                dic['data'] += [stored_data]

                if checkpoint_path:
                    append_inference_checkpoint(checkpoint_path, {
                        'idx': i,
                        'responses': r,
                        'data': stored_data,
                    })

                if len(ims_) >= 1:
                    for each in ims_:
                        try:
                            os.remove(each)
                        except OSError:
                            pass
            except Exception as e:
                print(e)
                pass

        self.results = dic
        return self.to_df(output=True)

    def to_df(self, output: bool = True) -> Any:
        """
            Convert the output from an MLLM reponse (from .batch_inference) into a DataFrame.

            Args:
                output (bool): Whether to return a DataFrame. Defaults to True.
            Returns:
                pd.DataFrame: A DataFrame containing responses and associated metadata.
        """

        if self.results is not None:
            df_list = []
            responses = self.results['responses']
            imgs = self.results['data']

            for inx in range(len(responses)):
                r = responses[inx]
                i = imgs[inx]

                r = pd.DataFrame(r['responses'])
                r = responses_to_wide_all_columns(r)
                for j in range(len(i)):
                    r[f'data_{j + 1}'] = i[j]

                df_list += [r]
            self.df = pd.concat(df_list, ignore_index=True)
            if output:
                return self.df
            return None
        else:
            return None

    def _mtmd(self,
              llm: str = None,
              mp: str = None,
              system_message: str = '',
              prompt: str = '',
              imgs: list = None,
              temperature: float = 0.2,
              top_k: int = 40,
              top_p: float = 0.9,
              min_p: float = 0.0,
              seed: int = 3407,
              ctx_size:int = 4096,
              max_new_tokens: int = 512,
              # threads:int = -1,
              # batch_size:int = 512,
              # gpu_layers:int = -1,
              schema = None,
              audio_input = False):
        '''

        Args:
            llm (str): model path
            mp (str):
            system_message (str, optional):
            prompt (str): prompt to start generation with
            imgs (list): list of image paths
            temperature (float): temperature (default: 0.2)
            top_k (float): top-k sampling (default: 40, 0 = disabled)
            top_p (float): top-p sampling (default: 0.9, 1.0 = disabled)
            min_p (float): min-p sampling (default: 0.0, 0.0 = disabled)
            seed (int): random seed
            ctx_size (int): size of the prompt context (default: 4096, 0 = loaded from model)
        '''
        # Security: subprocess is invoked with a list (shell=False, the default),
        # so each element is passed as a separate argv token to the OS — there is
        # no shell interpolation and no injection risk from the dynamic values
        # below.  All user-supplied paths go through pathlib.Path which rejects
        # null bytes; the prompt/system strings are positional arguments, not
        # shell metacharacters.  shlex.escape() is intentionally omitted because
        # it is only needed for shell=True invocations.
        if llm is None:
            raise ValueError("llm must not be None")
        if "\x00" in (system_message or "") or "\x00" in (prompt or ""):
            raise ValueError("Null byte detected in system_message or prompt")

        if imgs is not None:
            imgs = [Path(img) for img in imgs]
            imgs = [["--image" if not audio_input else "--audio", str(i)] for i in imgs]
            imgs = [item for sublist in imgs for item in sublist]

        cmd = ["llama-mtmd-cli",
               "-p", (system_message or "") + (prompt or "")
        ]

        if mp is not None:
            lm = Path(llm)
            mp = Path(mp)
            cmd = cmd + ["-m", str(lm), "--mmproj", str(mp)]
        else:
            cmd = cmd + ["-hf", str(llm)]

        if imgs is not None:
            cmd = cmd + imgs

        if schema is not None:
            cmd = cmd + ["-j", schema_json(schema, inline_refs=True)]

        cmd = cmd + ["--temp", f"{temperature}",
                     "--top-k", f"{top_k}",
                     "--top-p", f"{top_p}",
                     "-c", f"{ctx_size}",
                     "-s", f"{seed}",
                     "--min-p", f"{min_p}",
                     "-n", f"{max_new_tokens}",
                     # "-t", f"{threads}",
                     # "-ub", f"{batch_size}",
                     # "-ngl", f"{gpu_layers}"
                     ]

        env = None
        if self.model_dir is not None:
            import os
            env = os.environ.copy()
            env["HF_HUB_CACHE"] = self.model_dir

        try:
            res = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        except subprocess.CalledProcessError as e:
            print("===== STDERR =====")
            print(e.stderr)
            print("===== STDOUT =====")
            print(e.stdout)
            print("Return code:", e.returncode)
            print("Command:", e.cmd)
            raise
        raw = res.stdout
        return raw
