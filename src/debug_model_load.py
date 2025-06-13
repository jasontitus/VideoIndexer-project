import torch
from sentence_transformers import SentenceTransformer

def main():
    """
    A minimal script to debug model loading issues.
    Tries to load the specified SentenceTransformer model and reports success or failure.
    """
    model_name = "Qwen/Qwen3-Embedding-8B"
    print(f"--- Model Loading Debug Script ---")
    print(f"Attempting to load model: {model_name}")

    try:
        # Determine the device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"Using device: {device}")
        
        # Forcing CPU to see if the issue is MPS-specific
        # print("Forcing device to CPU for diagnostics...")
        # device = "cpu"

        model = SentenceTransformer(model_name, device=device)
        
        print("\n✅ SUCCESS: Model loaded successfully.")
        print(f"   - Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test an encoding
        print("\nTesting a sample sentence encoding...")
        try:
            test_embedding = model.encode(["This is a test sentence."])
            print(f"   - Encoding successful. Output shape: {test_embedding.shape}")
            print("✅ SUCCESS: Model appears to be working correctly.")
        except Exception as e:
            print(f"\n❌ ERROR: Model loaded but failed during encoding.")
            print(f"   - Error: {e}")

    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model '{model_name}'.")
        print(f"   - Error type: {type(e).__name__}")
        print(f"   - Error message: {e}")

if __name__ == "__main__":
    main() 