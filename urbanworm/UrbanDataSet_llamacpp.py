from .UrbanDataSet import UrbanDataSet
from .utils import llamacpp_vision, download_gguf

class UrbanDataSet_llamacpp(UrbanDataSet):
    '''
    Dataset class for urban imagery inference using MLLMs with llama_cpp.
    '''

    def __init__(self, local_llm:str = None, mp:str = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = local_llm
        self.mp = mp
    '''
    Add data
    
    Args:
        local_llm (str): path to model file (.gguf)
        mp (str): path to multimodal projector (mmproj) file (.gguf)
        **kwargs: image (str) and images (list)
    '''

    def pull(self, model:str = None, path:str = None, include:str = None)-> str:
        '''
        download multimodal (GGUF) models from Hugging Face Hub

        Args:
            model (str): model repository on Huggingface
            path (str): path to save model
            include (str): Glob patterns to match files to download

        Returns: Downloading or error message
        '''

        try:
            self.logger.log("Downloading multimodal model...")
            message = download_gguf(model, path, include)
        except Exception as e:
            message = e
        self.logger.log("These are ready-to-use models at the Hugging Face page of the ggml-org: https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc")
        self.logger.log("More GGUF models on Huggingface with vision capabilities can be found here: https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=gguf")
        return message

    def oneImgChat(self, prompt: str = None, **kwargs) -> str:
        '''
        Chat with MLLM model with one image.
        Args:
            prompt (str): prompt including instruction and questions:

        Returns: response from MLLM
        '''

        if (self.llm is None) or (self.mp is None):
            return {"llm": self.llm, "mp": self.mp}
        res = llamacpp_vision(self.llm, self.mp, [self.img], prompt)
        return res

    def loopImgChat(self, prompt: str = None, output_df=None,
                    disableProgressBar=None, **kwargs):
        '''
            Chat with MLLM model for each image in a list.
            Args:
                prompt (str): prompt including instruction and questions:

            Returns: response from MLLM
        '''

        if (self.llm is None) or (self.mp is None):
            return {"llm": self.llm, "mp": self.mp}
        from tqdm import tqdm

        dic = {'responses': [], 'img': []}
        for i in tqdm(range(len(self.imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            r = llamacpp_vision(self.llm, self.mp, self.imgs, prompt)
            dic['responses'] += [r]
            dic['img'] += [self.imgs[i]]
        if output_df:
            import pandas as pd
            return pd.DataFrame(dic)
        return dic
        