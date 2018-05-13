import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import abc
import glob
import json
from scipy import io
import h5py
import random
import re
import reg_proposer as rp

N_IMAGES = None
DIR = 'datasets/train'
ANNOTATION_FILE = "annotation/train/digitStruct.json"
NEG_OVERLAP_THD = 0.05
POS_OVERLAP_THD = 0.6
PATCH_SIZE = (32,32)
random.seed(111)

class ImgExtraction:    
    def __init__(self, region_proposer, annotator, overlap_calculator):
        """Overlap calculator """
        self._positive_samples = []
        self._negative_samples = []
        self._positive_labels = []
        self._negative_labels = []
        
        self._region_proposer = region_proposer
        self._annotator = annotator
        self._overlap_calculator = overlap_calculator

    
    def extract_patch(self, image_files, patch_size, positive_overlap_thd, negative_overlap_thd):
        
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(image_files)).start()
    
        for i, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
         
            # 1. detect regions
            candidate_regions = self._region_proposer.detect(image)
            candidate_patches = candidate_regions.get_patches(dst_size=patch_size)
            candidate_boxes = candidate_regions.get_boxes()
             
            # 2. load ground truth
            true_boxes, true_labels = self._annotator.get_boxes_and_labels(image_file)
            true_patches = rp.Regions(image, true_boxes).get_patches(dst_size=patch_size)
            
            # 3. calc overlap
            overlaps = self._overlap_calculator.calc_ious_per_truth(candidate_boxes, true_boxes)

            # 4. add patch to the samples            
            for i, label in enumerate(true_labels):
                samples = candidate_patches[overlaps[i,:]>positive_overlap_thd]
                labels_ = np.zeros((len(samples), )) + label
                self._positive_samples.append(samples)
                self._positive_labels.append(labels_)

            self._positive_samples.append(true_patches)
            self._positive_labels.append(true_labels)
            overlaps_max = np.max(overlaps, axis=0)
            self._negative_samples.append(candidate_patches[overlaps_max<negative_overlap_thd])           
            bar.update(i)

        bar.finish()

        negative_samples = np.concatenate(self._negative_samples, axis=0)    
        negative_labels = np.zeros((len(negative_samples), 1))
        positive_samples = np.concatenate(self._positive_samples, axis=0)    
        positive_labels = np.concatenate(self._positive_labels, axis=0).reshape(-1,1)

        samples = np.concatenate([negative_samples, positive_samples], axis=0)
        labels = np.concatenate([negative_labels, positive_labels], axis=0)
        return samples, labels

class FileSorter:
    def __init__(self):
        pass
    
    def sort(self, list_of_strs):
        list_of_strs.sort(key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s
    
    def _alphanum_key(self, s):
        """ Split string into strings and numbers"""
        return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]


class File(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, filename):
        pass
    
    @abc.abstractmethod
    def write(self, data, filename, write_mode="w"):
        pass
    
    def _check_directory(self, filename):
        directory = os.path.split(filename)[0]
        if directory != "" and not os.path.exists(directory):
            os.mkdir(directory)

class FileJson(File):
    def read(self, filename):
        """    #print n_train_files
        load json file
        """
        return json.loads(open(filename).read())
    
    def write(self, data, filename, write_mode="w"):
        self._check_directory(filename)        
        with open(filename, write_mode) as f:
            json.dump(data, f, indent=4)


class FileMat(File):
    def read(self, filename):
        """
        load mat file
        """
        return io.loadmat(filename)
    
    def write(self, data, filename, write_mode="w"):
        self._check_directory(filename)        
        io.savemat(filename, data)

class FileHDF5(File):
    def read(self, filename, db_name):
        db = h5py.File(filename, "r")
        np_data = np.array(db[db_name])
        db.close()        
        return np_data
    
    def write(self, data, filename, db_name, write_mode="a", dtype="float"):
        """
        Write data to hdf5 
        """
        self._check_directory(filename)        
        # todo : overwrite check
        db = h5py.File(filename, write_mode)
        dataset = db.create_dataset(db_name, data.shape, dtype=dtype)
        dataset[:] = data[:]
        db.close()


def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True, random_order=True):
    """
    list files in a directory matched in defined pattern.
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    
    FileSorter().sort(files)
        
    if n_files_to_sample is not None:
        if random_order:
            files = random.sample(files, n_files_to_sample)
        else:
            files = files[:n_files_to_sample]
    return files

class Annotation:
    
    def __init__(self, annotation_file):
        self._load_annotation_file(annotation_file)
    
class SVHN_Ann(Annotation):
    
    def get_boxes_and_labels(self, image_file):
        
        annotation = self._get_annotation(image_file)
        
        bbs = []
        labels = []
        
        for box in annotation["boxes"]:
            x1 = int(box["left"])
            y1 = int(box["top"])
            w = int(box["width"])
            h = int(box["height"])
    
            bb = (y1, y1+h, x1, x1+w)
            label = int(box["label"])
            
            bbs.append(bb)
            labels.append(label)
        return np.array(bbs), np.array(labels)
            
    def _load_annotation_file(self, annotation_file):
        self._annotations = FileJson().read(annotation_file)
    
    def _get_annotation(self, image_file):
        
        _, image_file = os.path.split(image_file)
        index = int(image_file[:image_file.rfind(".")])
        annotation = self._annotations[index-1]

        if annotation["filename"] != image_file:
            raise ValueError("Annotation file should be sorted!!!!")
        else:
            return annotation


if __name__ == "__main__":

    # Load data
    files = list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
    n_files = len(files)
    # use 80% of data for training
    n_train_files = int(n_files * 0.8)
    
    ImgExtraction = ImgExtraction(rp.MSER_proposer(), SVHN_Ann(ANNOTATION_FILE), rp.OverlapCalc())
    train_samples, train_labels = ImgExtraction.extract_patch(files[:n_train_files], PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)

    ImgExtraction = ImgExtraction(rp.MSER_proposer(), SVHN_Ann(ANNOTATION_FILE), rp.OverlapCalc())
    validation_samples, validation_labels = ImgExtraction.extract_patch(files[n_train_files:], PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)

    print train_samples.shape, train_labels.shape
    print validation_samples.shape, validation_labels.shape

    FileHDF5().write(train_samples, "projdata/train.hdf5", "images", "w", dtype="uint8")
    FileHDF5().write(train_labels, "projdata/train.hdf5", "labels", "a", dtype="int")
 
    FileHDF5().write(validation_samples, "projdata/val.hdf5", "images", "w", dtype="uint8")
    FileHDF5().write(validation_labels, "projdata/val.hdf5", "labels", "a", dtype="int")

