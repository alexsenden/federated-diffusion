import torch_fidelity

from .image_classes import plant_classes
from .tensor_only_wrapper import wrap_tensor_only


def calculate_metrics(gt, generated):
    return torch_fidelity.calculate_metrics(
        input1=wrap_tensor_only(generated),
        input2=wrap_tensor_only(gt),
        cuda=True,
        isc=True,
        fid=True,
    )


def write_metrics(plant_class, metrics, outfile):
    print(metrics)
    outfile.write(
        f'{plant_class},{metrics["inception_score_mean"]},{metrics["inception_score_std"]},{metrics["frechet_inception_distance"]}\n'
    )


# Images must already be transformed to 299x299 and normalized
def get_generative_metrics(gt, generated, output_dir):
    with open(f"{output_dir}/generative-metrics.csv", "w") as outfile:
        # Write header line
        outfile.write(
            "class,inception_score_mean,inception_score_std,frechet_inception_distance"
        )

        print(f"Calculating global metrics...")
        metrics = calculate_metrics(gt, generated)
        write_metrics("global", metrics, outfile)

        for plant_class in plant_classes:
            print(f"Processing {plant_class}...")
            generated_class_index = generated.class_to_idx[plant_class]
            generated_class_images = [
                img for img, label in generated if label == generated_class_index
            ]

            gt_class_index = gt.class_to_idx[plant_class]
            gt_class_images = [
                img for img, label in generated if label == gt_class_index
            ]

            print(f"Calculating {plant_class} metrics...")
            metrics = calculate_metrics(gt_class_images, generated_class_images)
