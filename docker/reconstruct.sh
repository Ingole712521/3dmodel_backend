#!/usr/bin/env bash
# Runs inside yeicor/colmap-openmvs with the scan directory mounted at /scan.
# Later stages may fail. The script still exits 0 when a mesh file exists.
set -uo pipefail

export PATH="/usr/local/bin/OpenMVS:${PATH}"

SCAN="${1:-/scan}"
IMAGES="$SCAN/images"
COLMAP_DIR="$SCAN/colmap"
MVS="$SCAN/openmvs"
OUT="$SCAN/output"
LOG="$SCAN/logs"
DB="$COLMAP_DIR/database.db"
SPARSE="$COLMAP_DIR/sparse"
DENSE="$COLMAP_DIR/dense"
MAIN_LOG="$LOG/reconstruction.log"
STAGE_CODE=1

mkdir -p "$COLMAP_DIR" "$SPARSE" "$MVS" "$OUT" "$LOG" "$IMAGES"
touch "$MAIN_LOG"

feature_extraction=false
feature_matching=false
sparse_reconstruction=false
dense_reconstruction=false
mesh_generation=false
mesh_refinement=false
texture_generation=false

note() {
  printf '%s\n' "$*" >> "$MAIN_LOG"
}

# OpenMVS stores image names relative to the directory that contains scene.mvs.
# The container working directory is /, so those names must be loaded with -w.
OMVS_WORK=(-w "$MVS")

inspect_frame() {
  perl - "$1" <<'PERL'
use strict;
use warnings;
my $path = $ARGV[0];
my $name = $path;
$name =~ s{.*/}{};
if (!-f $path) {
  print "INVALID\t$name\tsize=0\twidth=0\theight=0\tmime=missing\treason=missing\n";
  exit 1;
}
my $size = -s _;
if (!defined $size || $size <= 0) {
  print "INVALID\t$name\tsize=0\twidth=0\theight=0\tmime=application/octet-stream\treason=empty file\n";
  exit 1;
}
open my $fh, "<:raw", $path or do {
  print "INVALID\t$name\tsize=$size\twidth=0\theight=0\tmime=application/octet-stream\treason=unreadable\n";
  exit 1;
};
local $/;
my $data = <$fh>;
close $fh;
my ($width, $height, $mime) = (0, 0, "application/octet-stream");
if (substr($data, 0, 2) eq "\xFF\xD8") {
  $mime = "image/jpeg";
  my $i = 2;
  my $n = length($data);
  while ($i + 4 < $n) {
    if (ord(substr($data, $i, 1)) != 0xFF) { $i++; next; }
    my $marker = ord(substr($data, $i + 1, 1));
    while ($marker == 0xFF && $i + 2 < $n) { $i++; $marker = ord(substr($data, $i + 1, 1)); }
    $i += 2;
    next if $marker == 0xD8 || $marker == 0x01 || ($marker >= 0xD0 && $marker <= 0xD7);
    last if $marker == 0xD9 || $marker == 0xDA;
    last if $i + 2 > $n;
    my $length = unpack("n", substr($data, $i, 2));
    last if $length < 2 || $i + $length > $n;
    if ($marker >= 0xC0 && $marker <= 0xCF && $marker != 0xC4 && $marker != 0xC8 && $marker != 0xCC) {
      $height = unpack("n", substr($data, $i + 3, 2));
      $width = unpack("n", substr($data, $i + 5, 2));
      last;
    }
    $i += $length;
  }
  my $tail = substr($data, $n > 32 ? $n - 32 : 0);
  if ($width <= 0 || $height <= 0 || $tail !~ /\xFF\xD9/) {
    print "INVALID\t$name\tsize=$size\twidth=$width\theight=$height\tmime=$mime\treason=failed loading image header\n";
    exit 1;
  }
} elsif (length($data) >= 24 && substr($data, 0, 8) eq "\x89PNG\r\n\x1a\n") {
  $mime = "image/png";
  ($width, $height) = unpack("NN", substr($data, 16, 8));
  if ($width <= 0 || $height <= 0) {
    print "INVALID\t$name\tsize=$size\twidth=$width\theight=$height\tmime=$mime\treason=failed loading image header\n";
    exit 1;
  }
} else {
  print "INVALID\t$name\tsize=$size\twidth=0\theight=0\tmime=$mime\treason=invalid JPEG header\n";
  exit 1;
}
print "VALID\t$name\tsize=$size\twidth=$width\theight=$height\tmime=$mime\n";
exit 0;
PERL
}

verify_openmvs_images() {
  local dir="$1"
  local first=""
  local img line name rel
  shopt -s nullglob
  local files=("$dir"/*)
  shopt -u nullglob
  if [ "${#files[@]}" -eq 0 ]; then
    local sample=""
    sample=$(find "$IMAGES" -maxdepth 1 -type f -name 'frame_*.jpg' | sort | head -n 1)
    sample=$(basename "${sample:-unknown.jpg}")
    note "INVALID ${sample} size=0 width=0 height=0 mime=missing reason=undistorted image was not written"
    note "Corrupted frame detected: ${sample}"
    exit 1
  fi
  note "===== openmvs image check START ====="
  for img in "${files[@]}"; do
    [ -f "$img" ] || continue
    line=$(inspect_frame "$img" || true)
    name=$(printf '%s' "$line" | cut -f2)
    note "$line"
    rel="../colmap/dense/images/${name}"
    if ! ( cd "$MVS" && [ -s "$rel" ] ); then
      note "INVALID ${name} size=0 width=0 height=0 mime=missing reason=OpenMVS path ${rel} is not visible from ${MVS}"
      line="INVALID ${name}"
    fi
    if [[ "$line" == INVALID* ]]; then
      printf '%s\n' "$(printf '%s' "$line" | cut -f2- | tr '\t' ' ')" >> "$LOG/invalid_frames.txt"
      if [ -z "$first" ]; then
        first=$name
      fi
    fi
  done
  if [ -n "$first" ]; then
    note "Corrupted frame detected: ${first}"
    note "===== openmvs image check END failed ====="
    exit 1
  fi
  note "===== openmvs image check END ok files=${#files[@]} ====="
}

run_stage() {
  local key="$1"
  shift
  local stage_log="$LOG/${key}.log"
  local start end
  : > "$stage_log"
  note "===== ${key} START $(date -Is) ====="
  note "COMMAND: $*"
  start=$(date +%s)
  "$@" > >(tee -a "$MAIN_LOG" "$stage_log" >/dev/null) 2>&1
  STAGE_CODE=$?
  end=$(date +%s)
  note "===== ${key} END code=${STAGE_CODE} seconds=$((end - start)) ====="
}

write_status() {
  find "$SCAN" -type f | sed "s|^${SCAN}/||" | sort > "$LOG/files.txt"
  cat > "$LOG/status.json" <<EOF
{
  "feature_extraction": ${feature_extraction},
  "feature_matching": ${feature_matching},
  "sparse_reconstruction": ${sparse_reconstruction},
  "dense_reconstruction": ${dense_reconstruction},
  "mesh_generation": ${mesh_generation},
  "mesh_refinement": ${mesh_refinement},
  "texture_generation": ${texture_generation},
  "glb_export": false,
  "output_files": []
}
EOF
}

finish() {
  trap - EXIT
  write_status
  if find "$MVS" -maxdepth 1 -name '*.ply' -size +0c | grep -q .; then
    note "Mesh file exists. Reconstruction stage succeeded."
    exit 0
  fi
  note "No mesh file was written."
  exit 1
}

trap finish EXIT

run_stage feature_extraction colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.max_image_size 2000
if [ "$STAGE_CODE" -eq 0 ] && [ -s "$DB" ]; then
  feature_extraction=true
else
  exit 1
fi

run_stage feature_matching colmap sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 20 \
  --SequentialMatching.quadratic_overlap 1 \
  --FeatureMatching.use_gpu 1
if [ "$STAGE_CODE" -eq 0 ]; then
  feature_matching=true
else
  exit 1
fi

run_stage sparse_reconstruction colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --output_path "$SPARSE"

MODEL=""
BEST=0
for dir in "$SPARSE"/*; do
  [ -d "$dir" ] || continue
  size=0
  if [ -f "$dir/points3D.bin" ]; then
    size=$(stat -c%s "$dir/points3D.bin")
  elif [ -f "$dir/points3D.txt" ]; then
    size=$(stat -c%s "$dir/points3D.txt")
  fi
  if [ "$size" -ge "$BEST" ]; then
    BEST=$size
    MODEL=$dir
  fi
done
if [ -n "$MODEL" ]; then
  sparse_reconstruction=true
  note "Sparse model: $MODEL"
else
  note "COLMAP could not register the extracted frames."
  exit 1
fi

run_stage image_undistorter colmap image_undistorter \
  --image_path "$IMAGES" \
  --input_path "$MODEL" \
  --output_path "$DENSE" \
  --output_type COLMAP \
  --max_image_size 2000
if [ "$STAGE_CODE" -ne 0 ]; then
  exit 1
fi

verify_openmvs_images "$DENSE/images"

run_stage interface_colmap InterfaceCOLMAP "${OMVS_WORK[@]}" -i "$DENSE" -o "$MVS/scene.mvs" --image-folder "$DENSE/images/"
if [ "$STAGE_CODE" -ne 0 ] || [ ! -s "$MVS/scene.mvs" ]; then
  exit 1
fi

run_stage dense_reconstruction DensifyPointCloud "${OMVS_WORK[@]}" "$MVS/scene.mvs" --resolution-level 1
if [ "$STAGE_CODE" -eq 0 ] || [ -s "$MVS/scene_dense.mvs" ] || [ -s "$MVS/scene_dense.ply" ]; then
  dense_reconstruction=true
fi
if [ "$dense_reconstruction" != true ]; then
  exit 1
fi

DENSE_SCENE="$MVS/scene_dense.mvs"
if [ ! -s "$DENSE_SCENE" ]; then
  DENSE_SCENE="$MVS/scene.mvs"
fi

run_stage mesh_generation ReconstructMesh "${OMVS_WORK[@]}" "$DENSE_SCENE"
if [ "$STAGE_CODE" -eq 0 ] || [ -s "$MVS/scene_dense_mesh.ply" ]; then
  mesh_generation=true
fi
if [ "$mesh_generation" != true ]; then
  exit 1
fi

if [ -s "$MVS/scene_dense_mesh.mvs" ]; then
  run_stage mesh_refinement RefineMesh "${OMVS_WORK[@]}" "$MVS/scene_dense_mesh.mvs"
  if [ "$STAGE_CODE" -eq 0 ] || [ -s "$MVS/scene_dense_mesh_refine.ply" ]; then
    mesh_refinement=true
  fi
else
  note "===== mesh_refinement SKIPPED scene_dense_mesh.mvs is missing ====="
fi

TEXTURE_MESH="$MVS/scene_dense_mesh.ply"
if [ -s "$MVS/scene_dense_mesh_refine.ply" ]; then
  TEXTURE_MESH="$MVS/scene_dense_mesh_refine.ply"
fi
run_stage texture_generation TextureMesh "${OMVS_WORK[@]}" "$DENSE_SCENE" -m "$TEXTURE_MESH" -o "$OUT/room.glb" --export-type glb
if [ "$STAGE_CODE" -eq 0 ] || [ -s "$OUT/room.glb" ]; then
  texture_generation=true
fi

note "OpenMVS stages finished. mesh_generation=${mesh_generation} mesh_refinement=${mesh_refinement} texture_generation=${texture_generation}"
exit 0
