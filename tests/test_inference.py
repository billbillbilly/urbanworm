def test_import_inference_llama():
    from urbanworm.inference.llama import InferenceLlamacpp
    assert InferenceLlamacpp is not None