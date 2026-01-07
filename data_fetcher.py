# CELL 3: Sentinel Hub Authentication (SECURE)
print("🛰️ Sentinel Hub Setup")
print("Get credentials from: https://apps.sentinel-hub.com/dashboard/#/account/settings")
print("\n1. Sign up at https://www.sentinel-hub.com/")
print("2. Go to Dashboard > User Settings > OAuth clients")
print("3. Create OAuth client and copy credentials\n")

# Use getpass to hide input (won't show on screen or in notebook output)
from getpass import getpass

CLIENT_ID = getpass("Enter your Client ID (hidden): ")
CLIENT_SECRET = getpass("Enter your Client Secret (hidden): ")

# Install sentinelhub package
!pip install sentinelhub -q

from sentinelhub import SHConfig

# Configure
config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.save()

# Clear variables from memory for extra security
import gc
CLIENT_ID = None
CLIENT_SECRET = None
gc.collect()

print("\n✅ Sentinel Hub configured successfully!")
print("⚠️ Credentials are stored securely and not visible in notebook output")




# CELL 5: Download Images
print("🚀 Starting Sentinel-2 download...\n")

config = SHConfig()

# Test with first 30 images (remove .head(30) for full dataset)
train_subset = train_df.head(30)
train_success, train_failed = download_all_images_sentinel(
    train_subset,
    config,
    dataset_name="train",
    output_dir="images"
)

# Test images
test_subset = test_df.head(10)
test_success, test_failed = download_all_images_sentinel(
    test_subset,
    config,
    dataset_name="test",
    output_dir="images"
)

print(f"\n🎉 Download complete!")
print(f"Train: {train_success} images | Test: {test_success} images")

# Display samples
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sample Sentinel-2 Satellite Images', fontsize=16)

for i in range(min(6, len(train_subset))):
    property_id = train_subset['id'].iloc[i] if 'id' in train_subset.columns else i
    img_path = f"images/train_{property_id}.jpg"

    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax = axes[i//3, i%3]
        ax.imshow(img)
        price = train_subset['price'].iloc[i] if 'price' in train_subset.columns else 'N/A'
        ax.set_title(f"ID: {property_id}\nPrice: ${price:,.0f}" if price != 'N/A' else f"ID: {property_id}")
        ax.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Ready for EDA and modeling!")
