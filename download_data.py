"""
Download dataset using kagglehub
Standalone script to download the dataset before training
"""
import kagglehub
import os

def download_dataset():
    """Download the misinformation dataset from Kaggle"""
    print("=" * 60)
    print("KAGGLE DATASET DOWNLOAD")
    print("=" * 60)
    
    dataset_name = 'stevenpeutz/misinformation-fake-news-text-dataset-79k'
    
    print(f"\nDownloading: {dataset_name}")
    print("This may take a few minutes depending on your connection...")
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(dataset_name)
        
        print(f"\n✓ Download complete!")
        print(f"Dataset location: {path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirname, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename} ({size_mb:.2f} MB)")
        
        print("\n" + "=" * 60)
        print("READY TO TRAIN!")
        print("=" * 60)
        print("Run: python train.py")
        
        return path
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have kaggle credentials configured")
        print("2. Check: ~/.kaggle/kaggle.json exists")
        print("3. Visit: https://www.kaggle.com/docs/api")
        raise


if __name__ == "__main__":
    download_dataset()
