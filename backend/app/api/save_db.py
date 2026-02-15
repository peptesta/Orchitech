import os
import json
import io
import zipfile
from flask import Blueprint, request, send_file
from PIL import Image

save_bp = Blueprint('save_dataset', __name__)

@save_bp.route('/save_dataset', methods=['POST'])
def save_dataset():
    try:
        metadata = json.loads(request.form.get('metadata', '[]'))
        
        # Prendiamo i due valori base (es. 256 e 512)
        dim1 = request.form.get('resize_w')
        dim2 = request.form.get('resize_h')
        folder_organized = request.form.get('folder_organized') == 'true'

        uploaded_files = request.files.getlist('images_files')
        files_dict = {f.filename: f for f in uploaded_files}

        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in metadata:
                filename = item['filename']
                boxes = item['boxes']
                
                if filename not in files_dict:
                    continue
                
                img = Image.open(files_dict[filename].stream).convert("RGB")
                base_name = os.path.splitext(filename)[0]

                for idx, box in enumerate(boxes):
                    # 1. Ritaglio (xmin, ymin, xmax, ymax)
                    crop = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    c_w, c_h = crop.size

                    # 2. Scelta intelligente del formato (Smart Resize)
                    if dim1 and dim2:
                        d1, d2 = int(dim1), int(dim2)
                        
                        # Definiamo i due orientamenti possibili
                        # Formato A: (Larghezza x Altezza)
                        # Formato B: (Altezza x Larghezza)
                        target_a = (max(d1, d2), min(d1, d2)) # Landscape (es. 512x256)
                        target_b = (min(d1, d2), max(d1, d2)) # Portrait  (es. 256x512)

                        # Scegliamo il target che ha l'orientamento simile al crop
                        if c_h > c_w:
                            final_target = target_b # Crop verticale -> Target verticale
                        else:
                            final_target = target_a # Crop orizzontale -> Target orizzontale
                        
                        # Ridimensionamento mantenendo le proporzioni (Letterboxing)
                        # Per non perdere informazioni, usiamo thumbnail + sfondo nero
                        temp_crop = crop.copy()
                        temp_crop.thumbnail(final_target, Image.Resampling.LANCZOS)
                        
                        final_img = Image.new("RGB", final_target, (0, 0, 0))
                        paste_x = (final_target[0] - temp_crop.size[0]) // 2
                        paste_y = (final_target[1] - temp_crop.size[1]) // 2
                        final_img.paste(temp_crop, (paste_x, paste_y))
                    else:
                        final_img = crop

                    # 3. Nome e salvataggio
                    crop_name = f"{base_name}_{idx}.jpg"
                    path_in_zip = os.path.join(base_name, crop_name) if folder_organized else crop_name

                    buf = io.BytesIO()
                    final_img.save(buf, format="JPEG", quality=90)
                    zf.writestr(path_in_zip, buf.getvalue())

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='dataset.zip'
        )

    except Exception as e:
        return {"error": str(e)}, 500