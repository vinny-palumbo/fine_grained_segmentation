import sys
import numpy as np
import skimage

from config import Config
import model as modellib
import visualize

# list of fashion class names
class_names = ['BG', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 
                'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 
                'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 
                'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 
                'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 
                'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 
                'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 
                'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 
                'ruffle', 'sequin', 'tassel']


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    
    # Give the configuration a recognizable name
    NAME = "fashion"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46  # fashion has 46 classes
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def generate_image(images, molded_images, windows, results):
    results_final = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            model.unmold_detections(results[0][i], results[3][i], # detections[i], mrcnn_mask[i]
                                   image.shape, molded_images[i].shape,
                                   windows[i])
        results_final.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
        r = results_final[i]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    return results_final


def detect(sess, model, filename):
        
        # get image
        image = skimage.io.imread(filename)
        images = [image]
        
        # preprocessing
        molded_images, image_metas, windows = model.mold_inputs(images)
        anchors = model.get_anchors(molded_images[0].shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        results = \
            sess.run(None, {"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})

        # postprocessing
        results_final = generate_image(images, molded_images, windows, results)
        
        
if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect and segment fashion items in images.')
    parser.add_argument('--image', required=True,
                        metavar="path to image",
                        help='Image to detect fashion items on')

    args = parser.parse_args()

    # Validate arguments
    assert args.image, "Provide --image to detect fashion items"
    print("Image: ", args.image)
    
    # create config
    config = InferenceConfig()
    config.display()
    
    # create model
    model = modellib.MaskRCNN(mode="inference", model_dir="logs", config=config)
    
    # run with ONNXRuntime
    import onnxruntime
    model_file_name = './mrcnn.onnx'
    sess = onnxruntime.InferenceSession(model_file_name)
    detect(sess, model, args.image)
    
    
    
