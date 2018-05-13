import reg_proposer as rp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.models
from abc import ABCMeta, abstractmethod
from dataloader import list_files
from datatrainer import GrayPreproc

detect_model = "model_detection.hdf5"
model_clf = "model_clf.hdf5"

mean_value_for_detector = 107.524
mean_value_for_recognizer = 112.833

model_input_shape = (32,32,1)
DIR = 'datasets/train_detect'
VID_DIR = 'video'

def draw_contour(image, region):
    image_drawn = image.copy()
    cv2.drawContours(image_drawn, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
    return image_drawn
    
def draw_box(image, box, thickness=4):
    image_drawn = image.copy()
    y1, y2, x1, x2 = box
    cv2.rectangle(image_drawn, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return image_drawn


def plot_contours(img, regions):
    n_regions = len(regions)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
      
    for i, region in enumerate(regions):
        clone = img.copy()
        clone = draw_contour(clone, region)
        
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        clone = draw_box(clone, (y, y+h, x, x+w))
        
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        plt.title('Contours'), plt.xticks([]), plt.yticks([])
     
    plt.show()


def plot_bounding_boxes(img, bounding_boxes, titles=None):

    n_regions = len(bounding_boxes)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    for i, box in enumerate(bounding_boxes):
        clone = img.copy()
        clone = draw_box(clone, box)
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        if titles:
            plt.title("{0:.2f}".format(titles[i])), plt.xticks([]), plt.yticks([])
     
    plt.show()


def plot_images(images, titles=None):

    n_images = len(images)
    n_rows = int(np.sqrt(n_images)) + 1
    n_cols = int(np.sqrt(n_images)) + 1
    
    for i, img in enumerate(images):
        clone = img.copy()
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(img)
        if titles:
            plt.title("{0:.2f}".format(titles[i]))
        plt.xticks([]), plt.yticks([])
    plt.show()

class Classifier:
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self, patches):
        pass
    
class CNN_clf(Classifier):
    
    def __init__(self, model_file, preprocessor, input_shape=(32,32,1)):
        self._model = keras.models.load_model(model_file)
        self._preprocessor = preprocessor
        self.input_shape = input_shape

    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        patches_preprocessed = self._preprocessor.run(patches)
        probs = self._model.predict_proba(patches_preprocessed, verbose=0)
        return probs
    
class TrueBinaryClassifier(Classifier):
    """Classifier always predict true """
    def __init__(self, model_file=None, preprocessor=None, input_shape=None):
        self._model = None
        self._preprocessor = None
        self.input_shape = input_shape
    
    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        probs = np.zeros((len(patches), 2))
        probs[:, 1] = 1
        
        return probs


class NonMaxSuppressor:
    def __init__(self):
        pass
    
    def run(self, boxes, patches, probs, overlap_threshold=0.3):
        """
        Parameters:
            boxes (ndarray of shape (N, 4))
            patches (ndarray of shape (N, 32, 32, 1))
            probs (ndarray of shape (N,))
            overlap_threshold (float)
        
        Reference: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype="float")
        probs = np.array(probs)
     
        pick = []
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
     
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probs)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of
            # picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of the bounding box and the
            # smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have overlap greater than the
            # provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
            
        # return only the bounding boxes that were picked
        return boxes[pick].astype("int"), patches[pick], probs[pick]
    

class NumFinder:
    
    def __init__(self, classifier, recognizer, region_proposer):

        self._cls = classifier
        self._recognizer = recognizer
        self._region_proposer = region_proposer
        
    
    def run(self, image, threshold=0.7, do_nms=True, show_result=True, nms_threshold=0.3, order = 0):
        """
        Public function to run the NumFinder.
        """
        
        # 1. Get candidate patches
        candidate_regions = self._region_proposer.detect(image)
        patches = candidate_regions.get_patches(dst_size=self._cls.input_shape)
        
        # 3. Run pre-trained classifier
        probs = self._cls.predict_proba(patches)[:, 1]
    
        # 4. Thresholding
        bbs, patches, probs = self._get_thresholded_boxes(candidate_regions.get_boxes(), patches, probs, threshold)
    
        # 5. non-maxima-suppression
        if do_nms and len(bbs) != 0:
            bbs, patches, probs = NonMaxSuppressor().run(bbs, patches, probs, nms_threshold)
        
        if len(patches) > 0:
            probs_ = self._recognizer.predict_proba(patches)
            y_pred = probs_.argmax(axis=1)
        
        if show_result:
            for i, bb in enumerate(bbs):
                
                image = draw_box(image, bb, 2)
                
                y1, y2, x1, x2 = bb
                msg = "{0}".format(y_pred[i])
                cv2.putText(image, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
            
            if order != -1:
                cv2.imwrite('MSER+CNN_{}.png'.format(order), image)
        
        return bbs, probs, image


    def _get_thresholded_boxes(self, bbs, patches, probs, threshold):

        bbs = bbs[probs > threshold]
        patches = patches[probs > threshold]
        probs = probs[probs > threshold]
        return bbs, patches, probs


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(VID_DIR, filename), image)


def video_gen(video_name = 'house.mp4', fps = 40, frame_ids = [50, 200, 500], output_prefix = "final_proj_",
                            counter_init = 1):

    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.next()
    h, w, d = image.shape

    out_path = "video/{}{}".format(output_prefix, video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)


    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print "Processing fame {}".format(frame_num)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.next()

        frame_num += 1

    video_out.release()


def test():
    # 1. image files
    img_files = list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=None, random_order=False)

    preproc_for_detector = GrayPreproc(mean_value_for_detector)
    preproc_for_recognizer = GrayPreproc(mean_value_for_recognizer)

    char_detector = CNN_clf(detect_model, preproc_for_detector, model_input_shape)
    char_recognizer = CNN_clf(model_clf, preproc_for_recognizer, model_input_shape)
    
    digit_spotter = NumFinder(char_detector, char_recognizer, rp.MSER_proposer())
    
    for i, img_file in enumerate(img_files):
        # 2. image
        img = cv2.imread(img_file)
        
        digit_spotter.run(img, threshold=0.5, do_nms=True, nms_threshold=0.1, order = i)


if __name__ == "__main__":
    # 1. image files
    img_files = list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=None, random_order=False)

    preproc_for_detector = GrayPreproc(mean_value_for_detector)
    preproc_for_recognizer = GrayPreproc(mean_value_for_recognizer)

    char_detector = CNN_clf(detect_model, preproc_for_detector, model_input_shape)
    char_recognizer = CNN_clf(model_clf, preproc_for_recognizer, model_input_shape)
    
    digit_spotter = NumFinder(char_detector, char_recognizer, rp.MSER_proposer())
    
    for i, img_file in enumerate(img_files):
        # 2. image
        img = cv2.imread(img_file)
        
        digit_spotter.run(img, threshold=0.5, do_nms=True, nms_threshold=0.1, order = i)

    """
    #Generate videos

    video_name = 'house_1.mp4'
    fps = 40
    frame_ids = [50,250,500]
    output_prefix = "final_proj_"
    counter_init = 1
    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.next()
    h, w, d = image.shape

    out_path = "video/{}{}".format(output_prefix, video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)


    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print "Processing fame {}".format(frame_num)
        #image = cv2.GaussianBlur(image, (5,5),0)
        image = cv2.medianBlur(image, 5)
        image = digit_spotter.run(image, threshold=0.7, do_nms=True, nms_threshold=0.5, order = -1)[2]    

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.next()

        frame_num += 1

    video_out.release()
    """
