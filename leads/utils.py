# In leads/utils.py

import os
import zipfile
from django.conf import settings
import pandas as pd

def save_dataframe_to_excel(df, filename):
    """Saves a pandas DataFrame to an Excel file in the media directory."""
    # Define the directory where processed files will be stored
    output_dir = os.path.join(settings.MEDIA_ROOT, 'processed_files')
    # Create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Write the DataFrame to an Excel file
    df.to_excel(filepath, index=False, engine='xlsxwriter')
    return filepath

def create_zip_file(filenames_to_zip, zip_filename):
    """Creates a zip file from a list of files and then deletes the original files."""
    output_dir = os.path.join(settings.MEDIA_ROOT, 'processed_files')
    zip_filepath = os.path.join(output_dir, zip_filename)
    
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for filename in filenames_to_zip:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                # Add file to zip, using just the filename to avoid deep paths
                zipf.write(filepath, arcname=filename) 
                # Clean up the individual Excel file after adding it to the zip
                os.remove(filepath)
    return zip_filepath

