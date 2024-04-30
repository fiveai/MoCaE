import json
import pdb
import mmdet.datasets.coco as coco


convert_id_data = 0

if convert_id_data:
	# ID Data
	paths = [	'data/coco/annotations/instances_val2017.json',
				'data/objects365/annotations/generalod_natural_covariance_shift_val_all.json']

	image_prefix = ['coco/val2017/', 'objects365/val/']

	output_path = 'data/coco/annotations/general_od_id_test.json'

else:
	# OOD Data
	paths = [	'data/objects365/annotations/generalod_ood_general_val.json',
				'data/objects365/annotations/generalod_ood_general_train.json',
				'data/svhn/train/generalod_ood_svhn_train.json',
				'data/svhn/test/generalod_ood_svhn_test.json',
				'data/inaturalist/annotations/generalod_ood_inat_val.json']

	image_prefix = ['objects365/val/', 'objects365/train/', 'svhn/', 'svhn/', 'inaturalist/images/']

	output_path = 'data/coco/annotations/general_od_ood_test.json'


datasets = [coco.CocoDataset(path, [], test_mode=True) for path in paths]

out_data = {}
if convert_id_data:
	out_data['info'] = 'OD-Robust ID dataset'
else:
	out_data['info'] = 'OD-Robust OOD dataset'

out_data['licenses'] = datasets[0].coco.dataset['licenses']
out_data['images'] = []
out_data['annotations'] = []

img_id_ctr = 0
ann_id_ctr = 0

for i, ds in enumerate(datasets):
	for j in range(len(ds.coco.dataset['images'])):
		img_id =  ds.coco.dataset['images'][j]['id']
		ann_ids = ds.coco.get_ann_ids(img_ids=[img_id])
		anns = ds.coco.load_anns(ann_ids)

		img_info = ds.coco.dataset['images'][j] 
		img_info['id'] = img_id_ctr
		img_info['file_name'] = image_prefix[i] + img_info['file_name']
		out_data['images'].append(img_info)

		for ann in anns:
			ann_info = ann
			ann_info['id'] = ann_id_ctr
			ann_info['image_id'] = img_id_ctr
			# If it is OOD then set category id to 0
			if convert_id_data == 0:
				ann_info['category_id'] = 0

			out_data['annotations'].append(ann_info)
			ann_id_ctr += 1
		img_id_ctr += 1

out_data['categories'] = datasets[0].coco.dataset['categories']

with open(output_path, 'w') as outfile:
  json.dump(out_data, outfile)

out_dataset = coco.CocoDataset(output_path, [], test_mode=True)

