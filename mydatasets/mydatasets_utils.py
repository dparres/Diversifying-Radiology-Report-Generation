import torch
from nltk.tokenize import wordpunct_tokenize

# https://github.com/ysmiura/ifcc/blob/6c111dbdfe7ce9d3150a5ad90360584cfd2b8442/clinicgen/text/tokenizer.py#L24
# Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation.
def ifcc_clean_report(report):
    report = report.lower()
    return ' '.join(wordpunct_tokenize(report))

def vilmedic_collate(batch, multi_image=None):
    # vilmedic_collate only accepts list of tensors (list of images that are transformed)

    # Return one image
    if not multi_image or multi_image == 1:
        return {'images': torch.stack([s for s in batch]), "images_mask": None}

    # Return multiple image
    new_batch = []
    new_masks = []
    for sample in batch:
        sample_images = sample
        # Remove image to get to multi_image
        if len(sample_images) > multi_image:
            sample_images = sample_images[:multi_image]
        # Pad with zeros to get to multi_image
        if len(sample_images) < multi_image:
            first_image = sample_images[0]
            for _ in range(multi_image - len(sample_images)):
                sample_images.append(first_image.new_zeros(first_image.size()))
        # Stack
        sample_images = torch.cat([s.unsqueeze(dim=0) for s in sample_images], dim=0)
        sample_mask = (sample_images.sum(dim=(1, 2, 3)) != 0)
        new_batch.append(sample_images)
        new_masks.append(sample_mask)

    collated = {'images': torch.stack(new_batch)[0],
                'images_mask': torch.stack(new_masks)[0]}
    return collated