mkdir data/raw/lra

# Clone and unpack the LRA object.
# This can take a long time, so get comfortable.
rm -rf ./data/raw/lra/lra_release.gz ./data/raw/lra/lra_release  # Clean out any old datasets.
wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P ./data/raw/lra

# Add a progress bar because this can be slow.
# linux: sudo apt install pv
pv ./data/raw/lra/lra_release.gz | tar -zx -C ./data/raw/lra/
