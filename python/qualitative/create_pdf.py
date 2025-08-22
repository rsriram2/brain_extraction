import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load CSV and filter
csv_file = pd.read_csv('synthstrip/annotations_synth.csv')
filtered = csv_file[csv_file['3 - Holes'] == 'yes']

# Define image directories
folder1 = '/dcs05/ciprian/smart/mistie_3/results/image_ss_original'
folder2 = '/dcs05/ciprian/smart/mistie_3/results/image_ss'
folder3 = '/dcs05/ciprian/smart/mistie_3/results/image_ss_synth'

bad_files = []
output_pdf = "holes_image_comparison.pdf"

# Create the PDF
with PdfPages(output_pdf) as pdf:
    for idx, row in filtered.iterrows():
        filename = row['Filename']
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        path3 = os.path.join(folder3, filename)

        # Skip missing files
        if not (os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3)):
            print(f"🚫 Missing file: {filename}")
            bad_files.append(filename)
            continue

        # Try to open images safely
        try:
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            img3 = Image.open(path3)
        except UnidentifiedImageError:
            for img_path, folder in zip([path1, path2, path3], [folder1, folder2, folder3]):
                try:
                    Image.open(img_path)
                except UnidentifiedImageError:
                    print(f"❌ Unreadable image file: {filename} in folder: {folder}")
            bad_files.append(filename)
            continue

        # Plot side-by-side
        w, h = img1.size
        fig, axs = plt.subplots(1, 3, figsize=(w*3/100, h/100), dpi=300)
        axs[0].imshow(img1, cmap='gray', interpolation='none')
        axs[0].set_title(f"Left: V1", fontsize=40, pad=60)
        axs[0].axis('off')

        axs[1].imshow(img2, cmap='gray', interpolation='none')
        axs[1].set_title(f"Robust V2", fontsize=40, pad=60)
        axs[1].axis('off')
        
        axs[2].imshow(img3, cmap='gray', interpolation='none')
        axs[2].set_title(f"Right: SynthStrip", fontsize=40, pad=60)
        axs[2].axis('off')        

        fig.suptitle(f"Filename: {filename}", fontsize=50, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)

# Report problematic files
if bad_files:
    print("\n⚠️ The following files could not be processed:")
    for file in bad_files:
        print(f" - {file}")

print(f"\n✅ PDF created: {output_pdf}")
