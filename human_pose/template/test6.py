import yaml
with open("/home/ddl/git/human_pose_estimation/human_pose/template/conf/dataset/ch_dataset.yaml") as f:
     list_doc = yaml.load(f.read(), Loader=yaml.FullLoader)
list_doc['train']['batch_size'] = 16
list_doc['valid']['batch_size'] = 16

with open("/home/ddl/git/human_pose_estimation/human_pose/template/conf/dataset/ch_dataset.yaml", 'w') as f:
    yaml.dump(list_doc, f)

exec(open('/home/ddl/git/human_pose_estimation/human_pose/template/train.py').read())