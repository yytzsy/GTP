# Sentence Specified Dynamic Video Thumbnail Generation
by Yitian Yuan, Lin Ma and Wenwu Zhu

## Introduction

Sentence specified dynamic video thumbnail generation aims to dynamically select and concatenate video clips from an original video to generate one video thumbnail, which not only provides a concise preview of the original video but also semantically corresponds to the given sentence description.

![](https://github.com/yytzsy/GTP/blob/master/intro_model.PNG)

## Dataset

The 'dataset' folder contains the annotated dataset for sentence specified dynamic video thumbnail generation, as well as other auxiliary data. Concrete annotation process for the dataset can be found in our paper.

#### ./dataset/glove.840B.300d_dict.npy
* Glove word embeddings in our work, the original features can be downloaded at: https://nlp.stanford.edu/projects/glove/.

#### ./dataset/activitynet: ActivityNet Captions data
* (1) train.json: Video caption training annotations from the ActivityNet Captions dataset
* (2) val_merge.json: Video caption validation annotations from the ActivityNet Captions dataset. In the AcitivtyNet Captions dataset, there are two validation set (val_1.json and val_2.json), we merge the two validation sets into one as the val_merge.json.
* (3) video_info.pkl: It is a dictionary which contains video information in ActivityNet.
```
 	video_info = pkl.load(open(video_info.pkl,'r'))
	video_name_list = video_info.keys()
	one_video_name = video_name_list[0]
	[video_fps,video_total_frames,_,_] = video_info[one_video_name]
```

#### ./dataset/annotated_thumbnail: Video thumbnail annotations.
* (1) train_id.txt, val_id.txt, test_id.txt: The video id in our train, val and test split. All the video ids are from the ActivityNet Captions dataset.
* (2) anno_train.txt, anno_val.txt, anno_test.txt: Video thumbnail annotations. 
* Each line contains one annotation: 'Video id', 'Trimmed video segment', 'Video caption', 'Video thumbnail annotations', 'Consistency'.
	* 'Trimmed video segment': This field identifies the start and end timestamps of the trimmed video segment, and the timestamps are in accordance with the video caption.
	* 'Video caption': The video caption describes the video content for the 'Trimmed video segment'. All the video captions as well as their timestamps are from the ActivityNet Captions dataset.
	* 'Video thumbnail annotations': Video thumbnail annotations for the 'Trimmed video segment' based on the 'Video caption'. There are several annotations from different annotators. For example, '[[1, 2, 4, 5, 7], [10, 11, 12], [2, 3, 4, 5, 6], [4, 5, 6, 7, 9]]' means there are 4 annotations. The first annotation '[1, 2, 4, 5, 7]' means that 5 short clips are selected to compose the video thumbnail for the trimmed video segment. The 5 short clips are located at 0-2s, 2-4s, 6-8s, 8-10s, 12-14s in the trimmed video segment.
	* 'Consistency': The consistency (0~1) between different annotations. Higher value means that different annotations are consistent with each other, and the annotators reach a consensus towards the video thumbnail selection.

### Other auxiliary data
* ActivityNet Captions dataset: https://cs.stanford.edu/people/ranjaykrishna/densevid/
* ActivityNet features: http://activity-net.org/challenges/2016/download.html

## Model implementation
Code for data processing, model construction, and model training and testing.

#### ./src/data_prepare/generate_batch_data.py
* Data preprocessing for model training, testing and validation.  If run
```
python generate_batch_data.py
```
A folder './data' will be constructed. Three subdirs are in this folder, which contain the train, test, and validation h5py files, respectively. 

#### ./src/GTP (GTP_C, GTP_G, GTP_P)
* The GTP model as well as its variants implementation. Please see the paper for details.
* Model training:
```
python run_GTP.py --task train
```
* Model testing:
```
python run_GTP.py --task test
```

## Citation
```
@inproceedings{videoThumbnailGTP,
  title={Sentence Specified Dynamic Video Thumbnail Generation},
  author={Yitian Yuan, Lin Ma and Wenwu Zhu},
  booktitle={Proceedings of the 2019 ACM on Multimedia Conference},
  year={2019},
}
```






