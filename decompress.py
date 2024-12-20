import os
import pdal

def decompress_laz_to_las(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.laz'):
            laz_path = os.path.join(input_dir, filename)
            las_filename = os.path.splitext(filename)[0] + '.las'
            las_path = os.path.join(output_dir, las_filename)
            
            print(f"Decompressing {laz_path} to {las_path}...")
            
            pipeline = {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": laz_path
                    },
                    {
                        "type": "writers.las",
                        "filename": las_path,
                        "compression": "uncompressed" 
                    }
                ]
            }
            
            try:
                pipeline_json = pdal.Pipeline(json.dumps(pipeline))
                count = pipeline_json.execute()
                print(f"Successfully decompressed {filename}.\n")
            except Exception as e:
                print(f"Error decompressing {filename}: {e}\n")

if __name__ == "__main__":
    import json
    
    input_directory = 'data/fractal/train'
    output_directory = 'data/fractal/train/decompressed'
    
    decompress_laz_to_las(input_directory, output_directory)
