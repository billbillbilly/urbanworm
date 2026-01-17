from urbanworm.inference.llama import InferenceLlamacpp

if __name__ == "__main__":
    one = InferenceLlamacpp.one_inference()
    batch = InferenceLlamacpp.batch_inference()