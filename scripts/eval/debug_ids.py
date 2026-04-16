import json, pickle, sys
ROOT = '/home/haomingyang03/code/osrcir'
ann = json.load(open(ROOT+'/datasets/GENECIS/genecis/change_attribute.json'))
gd = pickle.load(open(ROOT+'/precomputed_cache/genecis/genecis_change_attribute_gallery.pkl','rb'))
a = ann[0]
tid = a['target']['image_id']
gids = [g['image_id'] for g in a['gallery']]
print('tid:', repr(tid), type(tid).__name__, flush=True)
print('gids[:3]:', [repr(x) for x in gids[:3]], type(gids[0]).__name__, flush=True)
print('cache[:3]:', [repr(x) for x in gd['ids'][:3]], type(gd['ids'][0]).__name__, flush=True)
print('cache total:', len(gd['ids']), flush=True)
cset = set(gd['ids'])
print('tid in cache:', tid in cset, flush=True)
print('str tid in str cache:', str(tid) in {str(x) for x in gd['ids']}, flush=True)

# check coco too
ann2 = json.load(open(ROOT+'/datasets/GENECIS/genecis/change_object.json'))
gd2 = pickle.load(open(ROOT+'/precomputed_cache/genecis/genecis_change_object_gallery.pkl','rb'))
a2 = ann2[0]
tid2 = a2['target']['val_image_id']
gids2 = [g['val_image_id'] for g in a2['gallery']]
print('\ncoco tid:', repr(tid2), type(tid2).__name__, flush=True)
print('coco gids[:3]:', [repr(x) for x in gids2[:3]], type(gids2[0]).__name__, flush=True)
print('coco cache[:3]:', [repr(x) for x in gd2['ids'][:3]], type(gd2['ids'][0]).__name__, flush=True)
print('coco tid in cache:', tid2 in set(gd2['ids']), flush=True)
