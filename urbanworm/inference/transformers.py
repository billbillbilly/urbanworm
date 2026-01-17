# from ..utils.utils import *
# from .Inference import Inference
# import outlines
# from outlines.inputs import Chat, Image, Audio
#
# class InferenceTrans(Inference):
#     '''
#     Constructor for vision inference using MLLMs with open_clip.
#     '''
#
#     def __init__(self, hf_model=None, hf_processor=None, **kwargs):
#         '''
#             Args:
#                 hf_model: Huggingface model to use.
#                 hf_processor: Huggingface processor to use.
#         '''
#
#         super().__init__(**kwargs)
#         self.hf_model = hf_model
#         self.hf_processor = hf_processor
#
#     def one_inference(self,
#                       system: str = '',
#                       prompt: str = '',
#                       image: str | list | tuple = None,
#                       audio: str | list | tuple = None,
#                       temp: float = 0.2,
#                       top_k: int = 20,
#                       top_p: float = 0.8,
#                       max_token: int = 128
#                       ):
#         img = None
#         clip = None
#         prompt = system + prompt
#         if image is not None:
#             img = image
#         else:
#             img = self.img
#         if isinstance(img, list) or isinstance(img, tuple):
#             if not isinstance(img[0], str):
#                 self.logger.warning("a list of images can only be a flatten list")
#         else:
#             img = [img]
#
#         # Create the Outlines model
#         hf_model = self.hf_model
#         hf_processor = self.hf_processor
#         do_sample = (temp is not None and temp > 0) or (top_p is not None) or (top_k is not None)
#         if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
#             hf_model.generation_config.do_sample = bool(do_sample)
#             if temp is not None:
#                 hf_model.generation_config.temperature = temp
#             if top_p is not None:
#                 hf_model.generation_config.top_p = top_p
#             if top_k is not None:
#                 hf_model.generation_config.top_k = top_k
#         # avoid pad_token warnings if needed
#         try:
#             tok = getattr(hf_processor, "tokenizer", None)
#             if tok is not None and getattr(hf_model.generation_config, "pad_token_id", None) is None:
#                 hf_model.generation_config.pad_token_id = tok.eos_token_id
#         except Exception:
#             pass
#         model = outlines.from_transformers(hf_model, hf_processor)
#
#         result = self._mtmd(model,
#                             prompt,
#                             images=img,
#                             max_token=max_token)
#         return result
#
#     def batch_inference(self,
#                         system: str = '',
#                         prompt: str = '',
#                         temp: float = 0.2,
#                         top_k: int = 20,
#                         top_p: float = 0.8,
#                         max_token: int = 128,
#                         disableProgressBar: bool = False):
#         from tqdm import tqdm
#         prompt = system + prompt
#         if self.batch_images is not None:
#             imgs = self.batch_images
#         else:
#             imgs = self.imgs
#         for i in tqdm(range(len(imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
#             img = self.imgs[i]
#             try:
#                 r = self._mtmd(prompt,
#                                images=img,
#                                max_token=max_token)
#             except Exception as e:
#                 self.logger.warning("batch_inference: image %d failed (%s). Continuing.", i, e)
#         return None
#
#
#     def _mtmd(self,
#               model = None,
#               prompt: str = None,
#               images: list|tuple = None,
#               audios: list|tuple = None,
#               max_token:int = 128):
#
#         if images is not None:
#             images = [Image(load_image_auto(img)) for img in images]
#             messages = Chat([
#                 {"role": "user", "content": prompt},
#                 {
#                     "role": "user",
#                     "content": images,
#                 }
#             ])
#         elif audios is not None:
#             audios = [Audio(audio) for audio in audios]
#             messages = Chat([
#                 {"role": "user", "content": prompt},
#                 {
#                     "role": "user",
#                     "content": audios,
#                 }
#             ])
#         else:
#             messages = Chat([{"role": "user", "content": prompt}])
#
#         res = model(messages,
#                     output_type=self.schema,
#                     max_new_tokens=max_token)
#         return self.schema.model_validate_json(res)