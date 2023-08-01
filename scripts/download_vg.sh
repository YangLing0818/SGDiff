VG_DIR=datasets/vg
mkdir -p $VG_DIR

wget https://visualgenome.org/static/data/dataset/objects.json.zip -O $VG_DIR/objects.json.zip
wget https://visualgenome.org/static/data/dataset/attributes.json.zip -O $VG_DIR/attributes.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip -O $VG_DIR/relationships.json.zip
wget https://visualgenome.org/static/data/dataset/object_alias.txt -O $VG_DIR/object_alias.txt
wget https://visualgenome.org/static/data/dataset/relationship_alias.txt -O $VG_DIR/relationship_alias.txt
wget https://visualgenome.org/static/data/dataset/image_data.json.zip -O $VG_DIR/image_data.json.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $VG_DIR/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $VG_DIR/images2.zip

unzip $VG_DIR/objects.json.zip -d $VG_DIR
unzip $VG_DIR/attributes.json.zip -d $VG_DIR
unzip $VG_DIR/relationships.json.zip -d $VG_DIR
unzip $VG_DIR/image_data.json.zip -d $VG_DIR
unzip $VG_DIR/images.zip -d $VG_DIR/images
unzip $VG_DIR/images2.zip -d $VG_DIR/images
