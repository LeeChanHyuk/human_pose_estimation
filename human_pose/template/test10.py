import yaml
with open("/home/ddl/git/human_pose_estimation/human_pose/template/conf/architecture/action_transformer.yaml") as f:
     list_doc = yaml.load(f.read(), Loader=yaml.FullLoader)
list_doc['mode'] = 0

with open("/home/ddl/git/human_pose_estimation/human_pose/template/conf/architecture/action_transformer.yaml", "w") as f:
    yaml.dump(list_doc, f)

exec(open('/home/ddl/git/human_pose_estimation/human_pose/template/train.py').read())