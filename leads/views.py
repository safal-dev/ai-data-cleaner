# leads/views.py

import io
import pandas as pd
import requests
from datetime import datetime, date
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, FileResponse
from django.contrib import messages
from django.conf import settings
import os
import chardet
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout

import google.generativeai as genai
import mimetypes
import xlsxwriter

from .models import InstructionSet, Profile
from django.contrib.auth.models import User
from .forms import UserRegisterForm
from django.shortcuts import render, redirect # Make sure redirect is imported
from django.urls import reverse # Make sure reverse is imported if you use it for redirects



# --- Helper function for admin check ---
def is_profile_admin(user):
    return user.is_authenticated and hasattr(user, 'profile') and user.profile.is_admin

# --- Authentication Views ---

def signup_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, 'Account created successfully! You are now logged in.')
            return redirect('dashboard') # Redirect to the dashboard after signup
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")
    else:
        form = UserRegisterForm()
    return render(request, 'leads/signup.html', {'form': form})

def signin_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('dashboard') # Redirect to the dashboard after signin
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'leads/signin.html', {'form': form})

def index(request):
    """
    Handles the landing page. Redirects authenticated users to their dashboard.
    """
    print(f"DEBUG: Entering index view. User: {request.user.username if request.user.is_authenticated else 'Anonymous'}")
    print(f"DEBUG: User is authenticated: {request.user.is_authenticated}")

    if request.user.is_authenticated:
        print("DEBUG: User is authenticated. Attempting to redirect to dashboard.")
        return redirect('dashboard') # Redirects to the URL named 'dashboard'
    
    print("DEBUG: User is NOT authenticated. Rendering landing page.")
    return render(request, 'leads/landing_page.html')

@login_required
def signout_view(request):
    auth_logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('index') # Redirect to the landing page after signout

# --- Helper Functions for Gemini ---

def format_gemini_prompt(df_input, user_instructions):
    """
    Formats the input for Gemini, including user instructions for text/tabular data.
    """
    if not user_instructions:
        return ""

    instruction_part = f"""
    {user_instructions}

    ---

    Here is the raw input data for you to process. This data may come from different sources (CSV, Excel files, or raw pasted text). Your task is to apply the instructions above to this data and return only the cleaned result in CSV format. Do not include any explanations, comments, or code block delimiters beyond the CSV content itself. Ensure the CSV headers exactly match the output columns you were instructed to produce.

    Raw data (as CSV for structured data, or concatenated text for unstructured):
    """
    data_part = df_input.to_csv(index=False)
    return instruction_part + data_part

def call_gemini_api(prompt,):
    """
    Sends the prompt to Gemini (text-only model) and returns the cleaned CSV text.
    Uses 'models/gemini-2.0-flash'.
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('models/gemini-2.0-flash')

    try:
        response = model.generate_content(prompt)

        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```csv") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```csv"):].rstrip("```").strip()
        elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```"):].rstrip("```").strip()
        
        return cleaned_text

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}. Please check your internet connection or API key.")


def call_gemini_vision_api(image_parts, user_instructions):
    """
    Sends images and text instructions to a Gemini Vision model and returns extracted data.
    image_parts: A list of dicts, e.g., [{"mime_type": "image/jpeg", "data": b"..."}]
    user_instructions: Text instructions for Gemini.
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)

    # Corrected: Using a multimodal model for vision tasks
    # 'gemini-1.5-pro' is a good choice for this as it supports vision.
    # 'gemini-2.0-flash' is not multimodal, so it won't work for images.
    model = genai.GenerativeModel('models/gemini-2.0-flash') # Corrected model for vision

    content_parts = []
    content_parts.extend(image_parts)
    content_parts.append({"text": user_instructions})

    try:
        response = model.generate_content(content_parts)

        cleaned_text = response.text.strip()
        # Apply more robust cleaning for vision model output
        # Remove common markdown code block fences
        if cleaned_text.startswith("```csv"):
            cleaned_text = cleaned_text[len("```csv"):].strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        
        # Split into lines for more granular cleaning
        output_lines = cleaned_text.splitlines()
        processed_lines = []
        for i, line in enumerate(output_lines):
            stripped_line = line.strip()
            
            # Skip markdown table separator lines (like |---|---|)
            if all(c == '-' or c == '|' or c.isspace() for c in stripped_line) and len(stripped_line) > 0:
                continue 

            # Attempt to detect and remove conversational intros (e.g., "Okay here's the information...")
            # This is a heuristic; direct AI instruction for raw CSV is best
            if i == 0 and not ',' in stripped_line and not stripped_line.startswith('|'):
                if "okay" in stripped_line.lower() or "here's the information" in stripped_line.lower():
                    # Skip this line if it appears to be a conversational intro
                    continue

            # If it's a pipe-delimited line, convert it to comma-delimited
            if stripped_line.startswith('|') and stripped_line.endswith('|'):
                parts = [p.strip() for p in stripped_line.strip('|').split('|')]
                valid_parts = [p for p in parts if p] # Remove empty parts from extra pipes
                if valid_parts:
                    processed_lines.append(','.join(valid_parts))
            else:
                # Assume it's already a CSV line or other text that should be included
                processed_lines.append(stripped_line)
        
        # Join the processed lines back into a single string
        final_cleaned_output = '\n'.join(processed_lines)
        
        # Final check if output is still empty after all cleaning
        if not final_cleaned_output.strip():
            raise RuntimeError("AI response was cleaned, but resulted in empty data.")
        
        return final_cleaned_output

    except Exception as e:
        raise RuntimeError(f"Gemini Vision API call failed: {e}. Please ensure your API key is correct and valid. If using the Generative AI library, ensure it's properly configured.")


# --- Django Views (MODIFIED) ---

# This is now purely the landing page view
def index(request):
    """
    Renders the marketing landing page (your new design).
    This view does not require authentication and passes minimal context.
    """
    return render(request, 'leads/index.html')

def common_process_data_context():
    """
    Helper function to get common data needed for both digital and physical processing pages.
    This includes all instruction sets and the default one.
    """
    instruction_sets = InstructionSet.objects.all().order_by('name')
    default_instruction = InstructionSet.objects.filter(is_default=True).first()
    return {
        'instruction_sets': instruction_sets,
        'default_instruction': default_instruction,
    }

def process_digital_data_view(request):
    """
    Handles the page for processing digital files (like Excel/CSV).
    """
    # Get common data
    context = common_process_data_context()

    # Add specific data for digital files
    context.update({
        'page_title_prefix': 'Digital',
        'file_type_description': 'Excel/CSV files',
        'file_input_name': 'digital_files', # How the files will be named when sent in the form
        'file_input_accept': '.csv, .xls, .xlsx, .txt', # What file types are allowed
        'submit_button_text': 'Clean Digital Data',
        'form_action_url': '/process-digital-data/', # The URL this form will send data to
        'data_type': 'digital', # Tells the JavaScript how to handle the files
    })

    # If the user submitted the form (uploaded files)
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('digital_files') # Get the uploaded files
        selected_instruction_id = request.POST.get('selected_instruction_id') # Get the selected instruction

        if not uploaded_files:
            messages.error(request, "Please upload at least one digital file.")
        else:
            # *** YOUR DIGITAL FILE PROCESSING LOGIC GOES HERE ***
            # For now, it just shows a success message.
            messages.success(request, f"Processing {len(uploaded_files)} digital file(s) with instruction ID: {selected_instruction_id}")
            # You'll replace the line above with code that actually cleans your digital data.

        # Render the page again, showing any messages
        return render(request, 'leads/process_data.html', context)

    # If it's a GET request (just loading the page)
    return render(request, 'leads/process_data.html', context)


def process_physical_data_view(request):
    """
    Handles the page for processing physical documents (like images).
    """
    # Get common data
    context = common_process_data_context()

    # Add specific data for physical files (images)
    context.update({
        'page_title_prefix': 'Physical (Image)',
        'file_type_description': 'images',
        'file_input_name': 'physical_images', # How the images will be named when sent in the form
        'file_input_accept': 'image/jpeg, image/png, image/gif, image/bmp, image/tiff, .webp', # Allowed image types
        'submit_button_text': 'Process Images',
        'form_action_url': '/process-physical-data/', # The URL this form will send data to
        'data_type': 'physical', # Tells the JavaScript how to handle the files
    })

    # If the user submitted the form (uploaded images)
    if request.method == 'POST':
        uploaded_images = request.FILES.getlist('physical_images') # Get the uploaded images
        selected_instruction_id = request.POST.get('selected_instruction_id') # Get the selected instruction

        if not uploaded_images:
            messages.error(request, "Please upload at least one image file.")
        else:
            # *** YOUR IMAGE PROCESSING LOGIC GOES HERE ***
            # For now, it just shows a success message.
            messages.success(request, f"Processing {len(uploaded_images)} image file(s) with instruction ID: {selected_instruction_id}")
            # You'll replace the line above with code that actually processes your images.

        # Render the page again, showing any messages
        return render(request, 'leads/process_data.html', context)

    # If it's a GET request (just loading the page)
    return render(request, 'leads/process_data.html', context)


# This is the new dashboard view for processing forms and quota
@login_required
def dashboard_view(request):
    """
    Renders the dashboard page with file upload, text area, and image upload.
    Displays quota information for both anonymous and authenticated users.
    """
    instruction_sets = []
    default_instruction = None
    
    # All the logic for fetching user-specific data now lives here
    profile, created = Profile.objects.get_or_create(user=request.user)
    profile.check_and_reset_quota()
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    default_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()
    
    current_quota = profile.monthly_quota - profile.cleans_this_month
    total_quota = profile.monthly_quota
    show_admin_link = profile.is_admin

    context = {
        'instruction_sets': instruction_sets,
        'default_instruction': default_instruction,
        'current_quota': current_quota,
        'total_quota': total_quota,
        'show_admin_link': show_admin_link,
    }
    return render(request, 'leads/dashboard.html', context)

@login_required
def process_digital_data_view(request):
    # Fetch all instructions for the current user to populate the modal
    # Filter by request.user if InstructionSet has a user foreign key
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    
    # Get the default instruction to pre-select it and display its name
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method == 'POST':
        profile = request.user.profile
        profile.check_and_reset_quota()

        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
            return redirect('dashboard') # Redirect to dashboard, or process_digital_data with an error

        pasted_text = request.POST.get('pasted_text', '')
        # IMPORTANT: Matches 'name="digital_files"' from process_digital_data.html
        uploaded_files = request.FILES.getlist('digital_files')

        # --- Instruction Selection Logic (Standardized for POST) ---
        selected_instruction_id = request.POST.get('selected_instruction_id')
        selected_instruction = None # Initialize
        if selected_instruction_id:
            try:
                # Use get_object_or_404 if instruction_set is required to exist
                selected_instruction = get_object_or_404(InstructionSet, pk=selected_instruction_id, user=request.user)
            except Exception: # Catch any error in fetching, fallback to default
                messages.warning(request, "Selected instruction not found or does not belong to you. Attempting to use default.")
        
        # If no explicit valid selection, try to get default
        if not selected_instruction:
            selected_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()

        # If still no instruction, user must set one
        if not selected_instruction:
            messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
            return redirect('manage_instructions') # Redirect to manage instructions page

        # Now use 'selected_instruction' for processing
        user_instructions = selected_instruction.instructions.strip()


        if not pasted_text and not uploaded_files:
            messages.error(request, "Please upload at least one file or paste some text data.")
            # Pass all instructions and the currently used/selected instruction back to the template
            context = {
                'instruction_sets': instruction_sets, # All available instruction sets
                'default_instruction': selected_instruction # The instruction that was attempted to be used
            }
            return render(request, 'leads/process_digital_data.html', context)

        combined_data = []

        for uploaded_file in uploaded_files:
            try:
                _, file_extension = os.path.splitext(uploaded_file.name)
                file_extension = file_extension.lower().lstrip('.')

                if file_extension == 'csv':
                    best_df = pd.DataFrame()
                    raw_file_content_bytes = uploaded_file.read()

                    detected_encoding = None
                    try:
                        result = chardet.detect(raw_file_content_bytes)
                        detected_encoding = result['encoding']
                        if detected_encoding:
                            detected_encoding = detected_encoding.lower()
                            if detected_encoding == 'ascii':
                                detected_encoding = 'utf-8'
                    except Exception as e:
                        print(f"DEBUG: chardet failed for '{uploaded_file.name}': {e}")
                        detected_encoding = None

                    encodings_to_try = []
                    if detected_encoding: encodings_to_try.append(detected_encoding)
                    if 'utf-8' not in encodings_to_try: encodings_to_try.append('utf-8')
                    if 'utf-16' not in encodings_to_try: encodings_to_try.append('utf-16')
                    if 'latin1' not in encodings_to_try: encodings_to_try.append('latin1')
                    if 'cp1252' not in encodings_to_try: encodings_to_try.append('cp1252')

                    delimiters_to_try = [',', ';', '\t', '|']
                    max_columns_found = 0

                    for encoding_attempt in encodings_to_try:
                        for delimiter_attempt in delimiters_to_try:
                            try:
                                decoded_csv_data = raw_file_content_bytes.decode(encoding_attempt)
                                temp_df = pd.read_csv(
                                    io.StringIO(decoded_csv_data),
                                    sep=delimiter_attempt,
                                    on_bad_lines='skip',
                                    header=0
                                )

                                if not temp_df.empty and temp_df.shape[1] > max_columns_found:
                                    best_df = temp_df
                                    max_columns_found = temp_df.shape[1]
                                    if max_columns_found > 1: # Found a good parse
                                        break
                            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                                continue
                            except Exception as e:
                                print(f"DEBUG: Error trying encoding '{encoding_attempt}' and delimiter '{delimiter_attempt}' for '{uploaded_file.name}': {e}")
                                continue
                        if max_columns_found > 1: # Break from outer loop if good parse found
                            break
                    
                    if best_df.empty or max_columns_found <= 1:
                        messages.error(request, f"Failed to parse CSV columns from '{uploaded_file.name}' after trying multiple encodings and delimiters. Please check the file's actual format and contents.")
                        continue
                    else:
                        df = best_df

                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == 'txt':
                    try:
                        text_content = uploaded_file.read().decode('utf-8')
                        df = pd.DataFrame([{'raw_input_data': text_content, '_source_file': uploaded_file.name}])
                    except Exception as e:
                        messages.error(request, f"Failed to read text file '{uploaded_file.name}': {e}")
                        continue
                else:
                    messages.warning(request, f"Skipping unsupported file type: {uploaded_file.name}")
                    continue

                if not df.empty:
                    if file_extension not in ['txt']: # Don't add source file if it's already in the raw_input_data column
                        df['_source_file'] = uploaded_file.name
                    combined_data.append(df)
                else:
                    messages.warning(request, f"File '{uploaded_file.name}' was empty or could not be read after parsing.")
            except Exception as e:
                messages.error(request, f"Failed to process '{uploaded_file.name}': {e}")
                continue

        if pasted_text:
            pasted_df = pd.DataFrame([{'raw_input_data': pasted_text, '_source_file': 'Pasted_Data'}]) # Changed from Demo_Leads to Data
            combined_data.append(pasted_df)

        if not combined_data:
            messages.error(request, "No valid data found after processing files and text.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction # The instruction that was attempted to be used
            }
            return render(request, 'leads/process_digital_data.html', context)

        combined_input_df = pd.concat(combined_data, ignore_index=True)
        combined_input_df = combined_input_df.fillna('')

        # Use the selected_instruction's content
        gemini_prompt = format_gemini_prompt(combined_input_df, user_instructions)

        try:
            cleaned_csv_output = call_gemini_api(gemini_prompt)
            
            if not cleaned_csv_output.strip():
                messages.warning(request, "Gemini returned an empty response. Please check your instructions and input data.")
                context = {
                    'instruction_sets': instruction_sets,
                    'default_instruction': selected_instruction
                }
                return render(request, 'leads/process_digital_data.html', context)

            response = HttpResponse(cleaned_csv_output, content_type='text/csv')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CleanedLeads.csv"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'

            request.user.profile.cleans_this_month += 1
            request.user.profile.save()

            return response

        except RuntimeError as e:
            messages.error(request, f"AI Processing Error: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)
        except Exception as e:
            messages.error(request, f"An unexpected error occurred during data processing: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)
    
    # --- This block handles the GET request for the page ---
    else:
        context = {
            'instruction_sets': instruction_sets,
            'default_instruction': default_instruction # This will be used to set initial value in HTML
        }
        return render(request, 'leads/process_digital_data.html', context)


@login_required
def process_physical_data_view(request): # Renamed from process_image
    # Fetch all instructions for the current user to populate the modal
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    
    # Get the default instruction to pre-select it and display its name
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method == 'POST':
        profile = request.user.profile
        profile.check_and_reset_quota()

        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
            return redirect('dashboard') # Redirect to dashboard, or process_physical_data with error

        # IMPORTANT: Matches 'name="physical_images"' from process_physical_data.html
        uploaded_images = request.FILES.getlist('physical_images')
        
        # --- Instruction Selection Logic (Standardized for POST) ---
        selected_instruction_id = request.POST.get('selected_instruction_id') # Changed from 'instruction_set' to 'selected_instruction_id'
        selected_instruction = None # Initialize
        if selected_instruction_id:
            try:
                selected_instruction = get_object_or_404(InstructionSet, pk=selected_instruction_id, user=request.user)
            except Exception: # Catch any error in fetching, fallback to default
                messages.warning(request, "Selected instruction not found or does not belong to you. Attempting to use default.")
        
        # If no explicit valid selection, try to get default
        if not selected_instruction:
            selected_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()

        # If still no instruction, user must set one
        if not selected_instruction:
            messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
            return redirect('manage_instructions') # Redirect to manage instructions page

        user_instructions = selected_instruction.instructions.strip()


        if not uploaded_images:
            messages.error(request, "Please upload at least one image file.")
            # Pass all instructions and the currently used/selected instruction back to the template
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)

        image_parts = []
        supported_image_types = ['image/jpeg', 'image/png', 'image/webp'] # As per your original code
        
        for uploaded_image in uploaded_images:
            mime_type, _ = mimetypes.guess_type(uploaded_image.name)
            if not mime_type or mime_type not in supported_image_types:
                messages.warning(request, f"Skipping unsupported image type: {uploaded_image.name} ({mime_type}). Only JPG, PNG, WebP are primarily supported.")
                continue
            
            try:
                image_data = uploaded_image.read()
                image_parts.append({
                    "mime_type": mime_type,
                    "data": image_data
                })
            except Exception as e:
                messages.error(request, f"Failed to read image '{uploaded_image.name}': {e}")
                continue

        if not image_parts:
            messages.error(request, "No valid image files were provided for processing.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)

        try:
            extracted_data_raw_output = call_gemini_vision_api(image_parts, user_instructions)

            if not extracted_data_raw_output.strip():
                messages.warning(request, "AI returned an empty response for image processing. Please check your instructions and image content.")
                context = {
                    'instruction_sets': instruction_sets,
                    'default_instruction': selected_instruction
                }
                return render(request, 'leads/process_physical_data.html', context)

            # Read CSV into DataFrame, specifying comma as delimiter
            df = pd.read_csv(io.StringIO(extracted_data_raw_output), sep=',')

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='ExtractedData')
            excel_buffer.seek(0)

            response = HttpResponse(excel_buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ExtractedImageData.xlsx"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'

            request.user.profile.cleans_this_month += 1
            request.user.profile.save()

            return response

        except RuntimeError as e:
            messages.error(request, f"AI Processing Error: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)
        except pd.errors.ParserError as e:
            messages.error(request, f"AI returned data in an unreadable CSV format. Please refine your instructions. (Error: {e}) Raw AI output might be: {extracted_data_raw_output[:200]}...") # Show beginning of problematic output
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)
        except Exception as e:
            messages.error(request, f"An unexpected error occurred during image processing: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)
    
    # --- This block handles the GET request for the page ---
    else:
        context = {
            'instruction_sets': instruction_sets,
            'default_instruction': default_instruction
        }
        return render(request, 'leads/process_physical_data.html', context)


# --- Instruction Set Management Views ---
@login_required
def manage_instructions(request):
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    return render(request, 'leads/manage_instructions.html', {'instruction_sets': instruction_sets})

@login_required
def create_instruction(request):
    form_name = ''
    form_instructions = ''
    form_is_default = False

    if request.method == 'POST':
        form_name = request.POST.get('name', '').strip()
        form_instructions = request.POST.get('instructions', '').strip()
        form_is_default = 'is_default' in request.POST

        if not form_name or not form_instructions:
            messages.error(request, "Name and Instructions cannot be empty.")
            return render(request, 'leads/instruction_form.html', {
                'form_title': 'Create New Instructions',
                'name': form_name,
                'instructions': form_instructions,
                'is_default': form_is_default,
            })

        if InstructionSet.objects.filter(user=request.user, name=form_name).exists():
            messages.error(request, f"An instruction set named '{form_name}' already exists for you. Please choose a different name.")
            return render(request, 'leads/instruction_form.html', {
                'form_title': 'Create New Instructions',
                'name': form_name,
                'instructions': form_instructions,
                'is_default': form_is_default,
            })

        instruction_set = InstructionSet(
            user=request.user,
            name=form_name,
            instructions=form_instructions,
            is_default=form_is_default
        )
        instruction_set.save()
        messages.success(request, f"Instruction set '{form_name}' created successfully!")
        return redirect('manage_instructions')

    return render(request, 'leads/instruction_form.html', {
        'form_title': 'Create New Instructions',
        'name': form_name,
        'instructions': form_instructions,
        'is_default': form_is_default,
    })


@login_required
def edit_instruction(request, pk):
    instruction_set = get_object_or_404(InstructionSet, pk=pk, user=request.user)

    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        instructions = request.POST.get('instructions', '').strip()
        is_default = 'is_default' in request.POST

        temp_instruction_set = InstructionSet(name=name, instructions=instructions, is_default=is_default, pk=pk)

        if not name or not instructions:
            messages.error(request, "Name and Instructions cannot be empty.")
            return render(request, 'leads/instruction_form.html', {
                'form_title': 'Edit Instructions',
                'instruction_set': temp_instruction_set,
                'name': name,
                'instructions': instructions,
                'is_default': is_default,
            })
        
        if InstructionSet.objects.filter(user=request.user, name=name).exclude(pk=pk).exists():
            messages.error(request, f"An instruction set named '{name}' already exists for you. Please choose a different name.")
            return render(request, 'leads/instruction_form.html', {
                'form_title': 'Edit Instructions',
                'instruction_set': temp_instruction_set,
                'name': name,
                'instructions': instructions,
                'is_default': is_default,
            })

        try:
            instruction_set.name = name
            instruction_set.instructions = instructions
            instruction_set.is_default = is_default
            instruction_set.save()
            messages.success(request, f"Instruction set '{name}' updated successfully!")
            return redirect('manage_instructions')
        except Exception as e:
            messages.error(request, f"Error updating instruction set: {e}")
            return render(request, 'leads/instruction_form.html', {
                'form_title': 'Edit Instructions',
                'instruction_set': instruction_set,
                'name': instruction_set.name,
                'instructions': instruction_set.instructions,
                'is_default': instruction_set.is_default,
            })
    else:
        return render(request, 'leads/instruction_form.html', {
            'form_title': 'Edit Instructions',
            'instruction_set': instruction_set,
            'name': instruction_set.name,
            'instructions': instruction_set.instructions,
            'is_default': instruction_set.is_default,
        })

@login_required
def delete_instruction(request, pk):
    instruction_set = get_object_or_404(InstructionSet, pk=pk, user=request.user)

    if instruction_set.is_default and InstructionSet.objects.filter(user=request.user, is_default=True).count() == 1:
        messages.error(request, "Cannot delete the last default instruction set. Please set another as default first or create a new one.")
        return redirect('manage_instructions')

    if request.method == 'POST':
        instruction_set.delete()
        messages.success(request, f"Instruction set '{instruction_set.name}' deleted successfully.")
        return redirect('manage_instructions')
    
    messages.error(request, "Invalid request for deletion.")
    return redirect('manage_instructions')

@login_required
def set_default_instruction(request, pk):
    instruction_set = get_object_or_404(InstructionSet, pk=pk, user=request.user)

    if request.method == 'POST':
        if instruction_set.is_default:
            messages.info(request, f"'{instruction_set.name}' is already the default instruction set.")
        else:
            # First, unset any other default for this user
            InstructionSet.objects.filter(user=request.user, is_default=True).update(is_default=False)
            instruction_set.is_default = True
            instruction_set.save()
            messages.success(request, f"'{instruction_set.name}' is now the default instruction set.")
    else:
        messages.error(request, "Invalid request to set default.")
    
    return redirect('manage_instructions')

# --- Admin User Management Views ---
@login_required
@user_passes_test(is_profile_admin)
def admin_user_management(request):
    users = User.objects.all().order_by('username')
    context = {
        'users': users,
    }
    return render(request, 'leads/admin_user_management.html', context)

@login_required
@user_passes_test(is_profile_admin)
def admin_edit_user(request, pk):
    user_to_edit = get_object_or_404(User, pk=pk)
    profile = user_to_edit.profile

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        monthly_quota = request.POST.get('monthly_quota', '').strip()
        is_admin_status = 'is_admin' in request.POST
        
        if not username or not email or not monthly_quota:
            messages.error(request, "Username, Email, and Monthly Quota cannot be empty.")
            return render(request, 'leads/admin_edit_user.html', {
                'user_to_edit': user_to_edit,
                'profile': profile,
                'form_username': username,
                'form_email': email,
                'form_monthly_quota': monthly_quota,
                'form_is_admin_status': is_admin_status,
                'pk': pk,
            })
        
        try:
            monthly_quota = int(monthly_quota)
            if monthly_quota < 0:
                messages.error(request, "Monthly Quota cannot be negative.")
                return render(request, 'leads/admin_edit_user.html', {
                    'user_to_edit': user_to_edit,
                    'profile': profile,
                    'form_username': username,
                    'form_email': email,
                    'form_monthly_quota': monthly_quota,
                    'form_is_admin_status': is_admin_status,
                    'pk': pk,
                })
        except ValueError:
            messages.error(request, "Monthly Quota must be a valid number.")
            return render(request, 'leads/admin_edit_user.html', {
                'user_to_edit': user_to_edit,
                'profile': profile,
                'form_username': username,
                'form_email': email,
                'form_monthly_quota': monthly_quota,
                'form_is_admin_status': is_admin_status,
                'pk': pk,
            })

        if User.objects.filter(username=username).exclude(pk=pk).exists():
            messages.error(request, f"Username '{username}' is already taken.")
            return render(request, 'leads/admin_edit_user.html', {
                'user_to_edit': user_to_edit,
                'profile': profile,
                'form_username': username,
                'form_email': email,
                'form_monthly_quota': monthly_quota,
                'form_is_admin_status': is_admin_status,
                'pk': pk,
            })
        if email != user_to_edit.email and User.objects.filter(email=email).exists():
            messages.error(request, f"Email '{email}' is already taken by another user.")
            return render(request, 'leads/admin_edit_user.html', {
                'user_to_edit': user_to_edit,
                'profile': profile,
                'form_username': username,
                'form_email': email,
                'form_monthly_quota': monthly_quota,
                'form_is_admin_status': is_admin_status,
                'pk': pk,
            })

        try:
            user_to_edit.username = username
            user_to_edit.email = email
            user_to_edit.save()

            profile.monthly_quota = monthly_quota
            profile.is_admin = is_admin_status
            profile.save() # Ensure save() is called after setting properties
            messages.success(request, f"User '{username}' updated successfully.")
            return redirect('admin_user_management')
        except Exception as e:
            messages.error(request, f"Error updating user: {e}")
            return render(request, 'leads/admin_edit_user.html', {
                'user_to_edit': user_to_edit,
                'profile': profile,
                'form_username': username,
                'form_email': email,
                'form_monthly_quota': monthly_quota,
                'form_is_admin_status': is_admin_status,
                'pk': pk,
            })
    else:
        return render(request, 'leads/admin_edit_user.html', {
            'user_to_edit': user_to_edit,
            'profile': profile,
            'form_username': user_to_edit.username, # Default values from existing user
            'form_email': user_to_edit.email,
            'form_monthly_quota': profile.monthly_quota,
            'form_is_admin_status': profile.is_admin,
            'pk': pk,
        })