from __future__ import annotations

from os import unlink

import ollama
from ollama import Client
from tqdm import tqdm
from ..utils.utils import *
from typing import Union
from .Inference import Inference
from .format import Response, schema_json
from .format import create_format


class InferenceOllama(Inference):
    '''
    Constructor for vision inference using MLLMs with Ollama.

    Args:
        llm (str): model checkpoint.
        ollama_key (str): The Ollama API key.
        **kwargs: image (str|list[str]|tuple[str]), images (list|tuple), data constructor (GeoTaggedData), and schema (dict)
    '''

    def __init__(self,
                 llm: str = None,
                 ollama_key: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.skip_errors = True
        self.ollama_key = ollama_key

    def one_inference(self,
                      system: str = '',
                      prompt: str = '',
                      image: str | list[str] | tuple[str] = None,
                      audio: str | list[str] | tuple[str] = None,
                      temp: float = 0.0,
                      top_k: int = 20.0,
                      top_p: float = 0.8):

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

        Notes:
            Ollama currently does not support audio input.
            The argument `audio` is just a placeholder for the future development.

        Returns:
            dict: A dictionary includes questions/messages, responses/answers
        '''

        ollama.pull(self.llm, stream=True)
        audio_input = False
        multiImg = False
        if image is None and audio is not None:
            image = audio
            audio_input = True
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
                        disableProgressBar: bool = False) -> dict:
        '''
        Chat with MLLM model for each image.

        Args:
            system (str, optinal): The system message.
            prompt (str): The prompt message.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
            disableProgressBar (bool): The progress bar for showing the progress of data analysis over the units.

        Returns:
            list A list of dictionaries. Each dict includes questions/messages, responses/answers, and image base64 (if required)
        '''

        ollama.pull(self.llm, stream=True)
        dic = {'responses': [], 'data': []}

        if self.batch_images is not None:
            imgs = self.batch_images
        else:
            imgs = self.imgs

        schema = create_format(self.schema)

        multiImgInput = False
        if isinstance(imgs[0], list) or isinstance(imgs[0], tuple):
            multiImgInput = True

        for i in tqdm(range(len(imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            img = imgs[i]
            try:
                r = self._mtmd(model=self.llm,
                               system=system, prompt=prompt,
                               img=img if multiImgInput else [img],
                               temp=temp, top_k=top_k, top_p=top_p,
                               schema=schema,
                               one_shot_lr=[],
                               multiImgInput=multiImgInput)
                rr = r.responses
            except Exception as e:
                # Log and continue; capture an error stub so downstream stays consistent
                self.logger.warning("batch_inference: image %d failed (%s). Continuing.", i, e)
                rr = {'error': str(e), 'data': None}

            dic['responses'] += [rr]
            dic['data'] += [imgs[i]]
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
              schema = None,
              one_shot_lr: list | tuple = [], multiImgInput: bool = False, audio_input: bool = False):

        if prompt is not None and img is not None:
            if len(img) == 1:
                return self._customized_chat(model, system, prompt, img[0], temp, top_k, top_p, schema, one_shot_lr)
            elif len(img) >= 2:
                system = f'You are analyzing aerial or street view images. For street view, you should just focus on the building and yard in the middle. {system}'
                if multiImgInput:
                    return self._customized_chat(model, system, prompt, img, temp, top_k, top_p, schema, one_shot_lr)
                else:
                    res = []

                    for i in range(len(img)):
                        r = self._customized_chat(model, system, prompt, img, temp, top_k, top_p, schema, one_shot_lr)
                        res += [r.responses]
                    return res
            return None
        else:
            raise Exception("Prompt or image(s) is missing.")

    def _customized_chat(self, model: str = None,
                         system: str = None, prompt: str = None, img: str | list | tuple = None,
                         temp: float = None, top_k: float = None, top_p: float = None,
                         schema=None,
                         one_shot_lr: list = [],
                         audio_input: bool = False) -> Response:

        if isinstance(one_shot_lr, list):
            if len(one_shot_lr) > 0:
                if not isinstance(one_shot_lr[0], dict):
                    raise Exception("Please provide a list of dictionaries.")

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
                    "top_p": top_p
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
                    "top_p": top_p
                }
            )

        raw_text = res.message.content
        try:
            return schema.model_validate_json(raw_text)
        except Exception:
            if self.skip_errors:
                raise
            else:
                pass

        repaired = sanitize_json_text(str(raw_text))
        try:
            return schema.model_validate_json(repaired)
        except Exception:
            pass

        extracted = extract_json_from_text(repaired) or repaired
        try:
            return schema.model_validate_json(extracted)
        except Exception:
            raise


import pandas as pd
from ..utils.utils import extract_last_json, responses_to_wide_all_columns
import subprocess
from pathlib import Path

class InferenceLlamacpp(Inference):
    '''
    Constructor for vision inference using MLLMs with llama.cpp

    Args:
        llm (str, optional): model checkpoint to download (e.g. ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0) or
        a local path to model file (.gguf)
        mp (str, optional): If `llm` is provided as a local path to model file (.gguf),
        `mp` has to be provided as a local path to multimodal projector file (*mproj*.gguf).
        **kwargs: image (str|list[str]|tuple[str]), images (list|tuple), data constructor (GeoTaggedData), and schema (dict)
    '''

    def __init__(self, llm:str = None, mp:str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.mp = mp

    def one_inference(self,
                      system: str = '',
                      prompt: str = '',
                      image: str | list | tuple = None,
                      audio: str | list | tuple = None,
                      temp: float = 0.2,
                      top_k: int = 20,
                      top_p: float = 0.8,
                      ctx_size: int = 4096,
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
                    temp = base64img2temp(i)
                    im_ += [temp]
                elif is_url(i):
                    temp = url2temp(i)
                    im_ += [temp]
                else:
                    pass
        else:
            for i in range(len(im)):
                if is_url(im[i]):
                    temp = sound_url_to_temp(im[i])
                    im_ += [temp]
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
                except:
                    pass
        return df

    def batch_inference(self,
                        system: str = '',
                        prompt: str = '',
                        temp: float = 0.2,
                        top_k: int = 20,
                        top_p: float = 0.8,
                        ctx_size: int = 4096,
                        audio_input = False,
                        disableProgressBar: bool = False):
        '''
            Chat with MLLM model for each image in a list.
            Args:
                system (str, optional): The system message.
                prompt (str): The prompt message.
                temp (float): The temperature value.
                top_k (float): The top_k value.
                top_p (float): The top_p value.
                ctx_size (int): Size of context (The default is 4096)
                audio_input (bool): Whether to run inference with audio input
                disableProgressBar (bool): Whether to disable progress bar.
            Returns: response from MLLM as a dataframe
        '''

        dic = {'responses': [], 'data': []}
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

        for i in tqdm(range(len(imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            ims = [imgs[i]] if isinstance(imgs[i], str) else imgs[i]

            ims_origin = None
            ims_ = []
            if not audio_input:
                for im in ims:
                    if is_base64(im):
                        temp = base64img2temp(im)
                        ims_ += [temp]
                    elif is_url(im):
                        temp = url2temp(im)
                        ims_ += [temp]
                    else:
                        pass
            else:
                for j in range(len(ims)):
                    im = ims[j]
                    if is_url(im):
                        if clips is not None:
                            clip = clips[j]
                            temp = sound_url_to_temp(im, clip)
                            ims_ += [temp]
                        else:
                            temp = sound_url_to_temp(im)
                            ims_ += [temp]
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
                                   ctx_size=ctx_size,
                                   schema=schema,
                                   audio_input = audio_input)
                    r = extract_last_json(r)
                    try_times += 1

                if r is None:
                    r = 'Bad response'
                dic['responses'] += [r]
                dic['data'] += [ims] if ims_origin is None else [ims_origin]

                if len(ims_) >= 1:
                    for each in ims_:
                        try:
                            os.remove(each)
                        except:
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
              ctx_size:int = 4096,
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
            ctx_size (int): size of the prompt context (default: 4096, 0 = loaded from model)
        '''
        if imgs is not None:
            imgs = [Path(img) for img in imgs]
            imgs = [["--image" if not audio_input else "--audio", str(i)] for i in imgs]
            imgs = [item for sublist in imgs for item in sublist]

        cmd = ["llama-mtmd-cli",
               "-p", system_message + prompt
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
                     # "-t", f"{threads}",
                     # "-ub", f"{batch_size}",
                     # "-ngl", f"{gpu_layers}"
                     ]

        try:
            res = subprocess.run(cmd, check=True, text=True, capture_output=True)
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
