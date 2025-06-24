# leads/views.py

import io
import pandas as pd
import requests
from datetime import datetime, date, timedelta # Added timedelta for date range calculations
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, FileResponse
from django.contrib import messages
from django.conf import settings
import os
import chardet
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.utils import timezone # For timezone-aware datetimes

import google.generativeai as genai
import mimetypes
import xlsxwriter
from decimal import Decimal # Crucial for precise currency calculations

from .models import InstructionSet, Profile, TransactionRecord # Imported TransactionRecord
from django.contrib.auth.models import User
from .forms import UserRegisterForm
# Note: 'render' and 'redirect' are already imported from django.shortcuts,
# so the lines below are redundant but harmless.
# from django.shortcuts import render, redirect 
# from django.urls import reverse # Already used, but keeping for clarity if standalone import was intended

from django.db.models import Sum # For dashboard aggregation


# --- Pricing Model Constants ---
# Per 1 Million tokens (USD)
PRICING_MODEL = {
    'gemini-2.0-flash': { # Model used for text processing (digital data)
        'input_cost_200k_threshold': 1.25,  # Cost per 1M tokens for prompts <= 200k
        'input_cost_over_200k': 2.50,       # Cost per 1M tokens for prompts > 200k
        'output_cost_200k_threshold': 10.00, # Cost per 1M tokens for output <= 200k
        'output_cost_over_200k': 15.00,      # Cost per 1M tokens for output > 200k
    },
    'gemini-1.5-pro-latest': { # Model used for vision processing (physical data)
        'input_cost_200k_threshold': 3.50,  # Example: adjust based on actual 1.5 Pro pricing
        'input_cost_over_200k': 7.00,
        'output_cost_200k_threshold': 10.50, # Example: adjust based on actual 1.5 Pro pricing
        'output_cost_over_200k': 21.00,
    }
}

TOKEN_THRESHOLD = 200000 # 200k tokens for pricing tier change
MILLION_TOKENS = 1_000_000 # Denominator for cost calculation (per 1 Million tokens)


# --- Helper function for admin check (not directly used in views, but good to have) ---
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
    # print(f"DEBUG: Entering index view. User: {request.user.username if request.user.is_authenticated else 'Anonymous'}")
    # print(f"DEBUG: User is authenticated: {request.user.is_authenticated}")

    if request.user.is_authenticated:
        # print("DEBUG: User is authenticated. Attempting to redirect to dashboard.")
        return redirect('dashboard') # Redirects to the URL named 'dashboard'
    
    # print("DEBUG: User is NOT authenticated. Rendering landing page.")
    return render(request, 'leads/index.html') # Assuming 'landing_page.html' is the marketing landing page

@login_required
def signout_view(request):
    auth_logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('index') # Redirect to the landing page after signout

# --- Helper Functions for Gemini API Interaction ---

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


def call_gemini_api(prompt):
    """
    Sends the prompt to Gemini (text-only model) and returns the cleaned CSV text,
    along with input/output token counts.
    Uses 'models/gemini-2.0-flash'.
    Returns: tuple (cleaned_text, total_input_tokens, total_output_tokens)
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model_name = 'models/gemini-2.0-flash' # The text processing model
    model = genai.GenerativeModel(model_name)

    total_input_tokens = 0
    total_output_tokens = 0

    try:
        response = model.generate_content(
            prompt,
            safety_settings=[ # Recommended safety settings for API calls
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="text/plain", # Request plain text for easier parsing
            )
        )

        # Access token usage from usage_metadata, which comes with the response object
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_input_tokens = response.usage_metadata.prompt_token_count
            total_output_tokens = response.usage_metadata.candidates_token_count

        cleaned_text = response.text.strip()
        # Robust cleaning to remove common markdown code block fences that Gemini might include
        if cleaned_text.startswith("```csv") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```csv"):].rstrip("```").strip()
        elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```"):].rstrip("```").strip()
        
        return cleaned_text, total_input_tokens, total_output_tokens

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}. Please check your internet connection or API key.")


def call_gemini_vision_api(image_parts, user_instructions):
    """
    Sends images and text instructions to a Gemini Vision model and returns extracted data
    as a clean CSV string, along with token counts.
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    # Use a model that supports vision, like 'gemini-1.5-flash'
    model_name = 'models/gemini-2.0-flash'
    model = genai.GenerativeModel(model_name)

    # --- START OF THE FIX: Create a more detailed prompt ---

    # Combine the user's simple instruction with our detailed formatting rules.
    full_system_prompt = f"""
    You are an expert data extraction assistant. Your task is to analyze the provided image(s) and extract the specific information requested by the user.

    USER'S INSTRUCTION: "{user_instructions}"

    YOUR TASK:
    1.  Analyze the image(s) to find the data requested in the user's instruction.
    2.  Format the extracted data as a raw CSV (Comma-Separated Values).
    3.  The first line of the CSV must be the header row. The header names should be based on the user's instruction (e.g., "Name,Email,Phone Number").
    4.  Each subsequent row should contain the data for one extracted entity.
    5.  Do NOT include any introductory text, explanations, or markdown formatting (like ```) in your response. Only return the raw CSV data.
    """

    # --- END OF THE FIX ---

    content_parts = []
    content_parts.extend(image_parts) # Add the image data
    content_parts.append({"text": full_system_prompt}) # Add our new, detailed prompt

    try:
        response = model.generate_content(content_parts)

        # Token count logic remains the same
        total_input_tokens = 0
        total_output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_input_tokens = response.usage_metadata.prompt_token_count
            total_output_tokens = response.usage_metadata.candidates_token_count

        # The robust text cleaning you already have is still good to keep as a fallback
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```csv") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```csv"):].rstrip("```").strip()
        elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```"):].rstrip("```").strip()
        
        # Now, the 'cleaned_text' should be in the correct CSV format from the start
        return cleaned_text, total_input_tokens, total_output_tokens

    except Exception as e:
        # It's better to return a failure indicator than to raise an exception here,
        # so the view can handle it gracefully.
        print(f"Gemini Vision API Error: {e}")
        error_message = f"Gemini Vision API call failed: {e}."
        # We will return None for the text to signal failure to the view.
        return None, 0, 0

def calculate_gemini_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the estimated cost based on Gemini's pricing model.
    model_name: The internal name of the model (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro-latest')
    input_tokens: Number of input tokens.
    output_tokens: Number of output tokens.
    Returns: decimal.Decimal cost (explicitly ensure Decimal type for financial precision)
    """
    pricing = PRICING_MODEL.get(model_name)
    if not pricing:
        print(f"Warning: Pricing for model '{model_name}' not found. Cost will be 0.")
        return Decimal('0.00') # Always return a Decimal type for consistency

    cost = 0.0 # Use float for intermediate calculation precision

    # Calculate input cost based on token threshold
    if input_tokens <= TOKEN_THRESHOLD:
        cost += (input_tokens / MILLION_TOKENS) * pricing['input_cost_200k_threshold']
    else:
        cost += (input_tokens / MILLION_TOKENS) * pricing['input_cost_over_200k']

    # Calculate output cost based on token threshold
    if output_tokens <= TOKEN_THRESHOLD:
        cost += (output_tokens / MILLION_TOKENS) * pricing['output_cost_200k_threshold']
    else:
        cost += (output_tokens / MILLION_TOKENS) * pricing['output_cost_over_200k']
    
    # CRITICAL: Convert the final 'cost' (which might be a float from intermediate math)
    # to a Decimal using its string representation for accuracy, then quantize it.
    return Decimal(str(cost)).quantize(Decimal('0.000001'))


# --- Django Views (User Facing) ---


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
# In your leads/views.py file

@login_required
def process_digital_data_view(request):
    """
    Handles the page for processing digital files (Excel/CSV/Text) and pasted text.
    - Validates user has instructions.
    - Processes multiple data sources.
    - Tracks token usage and cost.
    - Returns a downloadable CSV on success.
    - Redirects to a safe page on failure.
    """
    # 1. Get all instruction sets for the user.
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')

    # 2. CRITICAL CHECK: Redirect if no instructions exist (for GET and POST).
    if not instruction_sets.exists():
        messages.info(request, "Welcome! Please create your first AI instruction set to get started.")
        return redirect('manage_instructions') # URL name for your instruction creation page

    # Find the default instruction to pre-fill the form.
    default_instruction = instruction_sets.filter(is_default=True).first()

    # --- Start of POST request logic ---
    if request.method == 'POST':
        # Get user profile and check quota
        profile = request.user.profile
        profile.check_and_reset_quota()
        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}).")
            return redirect('dashboard')

        # Get instruction set from form
        selected_instruction_id = request.POST.get('selected_instruction_id')
        selected_instruction = instruction_sets.filter(pk=selected_instruction_id).first()
        if not selected_instruction:
            # If nothing was selected, fall back to default or the very first one available
            selected_instruction = default_instruction or instruction_sets.first()
        user_instructions = selected_instruction.instructions.strip()

        # --- Data Gathering and Parsing Logic ---
        pasted_text = request.POST.get('pasted_text', '').strip()
        uploaded_files = request.FILES.getlist('digital_files')

        if not pasted_text and not uploaded_files:
            messages.error(request, "Please upload at least one file or paste some text data.")
            return redirect('process_digital_data') # Redirect back to the same page to show the form again

        combined_data = [] # This will hold DataFrames from all sources

        # Process uploaded files
        for uploaded_file in uploaded_files:
            try:
                # This is a summary of your excellent, detailed parsing logic.
                # Your original code for this part is great and should be used here.
                df = None
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension == '.csv':
                    # ... your robust csv parsing logic ...
                    df = pd.read_csv(uploaded_file) # Simplified for this example
                elif file_extension in ['.xls', '.xlsx']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == '.txt':
                    text_content = uploaded_file.read().decode('utf-8')
                    df = pd.DataFrame([{'raw_input_data': text_content}])
                
                if df is not None and not df.empty:
                    df['_source_file'] = uploaded_file.name
                    combined_data.append(df)
            except Exception as e:
                messages.error(request, f"Failed to process file '{uploaded_file.name}': {str(e)}")
                continue # Safely skip to the next file

        # Process pasted text
        if pasted_text:
            try:
                # Attempt to read pasted text as if it were a CSV file in memory
                pasted_df = pd.read_csv(io.StringIO(pasted_text))
                pasted_df['_source_file'] = 'Pasted_Data'
                combined_data.append(pasted_df)
            except Exception:
                # If it fails, treat it as a single block of raw text
                pasted_df = pd.DataFrame([{'raw_input_data': pasted_text, '_source_file': 'Pasted_Data'}])
                combined_data.append(pasted_df)

        # Final check if any data was successfully processed
        if not combined_data:
            messages.error(request, "No valid data could be read from your files or pasted text.")
            return redirect('process_digital_data')

        # Combine all data sources into a single DataFrame for the AI
        combined_input_df = pd.concat(combined_data, ignore_index=True).fillna('')
        gemini_prompt = format_gemini_prompt(combined_input_df, user_instructions)

        # --- AI Call and Response Generation ---
        try:
            # Ensure your call_gemini_api returns (text, in_tokens, out_tokens) on success
            # and (None, 0, 0) on failure.
            cleaned_csv_output, input_tokens, output_tokens = call_gemini_api(gemini_prompt)

            if cleaned_csv_output is None:
                messages.error(request, "AI processing failed. This could be due to a network issue or an invalid API key. Please try again.")
                return redirect('dashboard') # Redirect to a safe page

            if not cleaned_csv_output.strip():
                messages.warning(request, "The AI returned an empty response. Please try adjusting your instructions.")
                return redirect('dashboard') # Redirect to a safe page

            # --- Costing, Profile Update, and Transaction Logging ---
            model_used = 'gemini-1.5-flash'
            cost = calculate_gemini_cost(request.user, model_used, input_tokens, output_tokens)

            profile.total_input_tokens += input_tokens
            profile.total_output_tokens += output_tokens
            profile.total_cost_usd += cost
            profile.cleans_this_month += 1
            profile.save()

            TransactionRecord.objects.create(
                user=request.user,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                transaction_type='digital',
            )

            # --- Create and return the downloadable CSV file ---
            response = HttpResponse(cleaned_csv_output, content_type='text/csv')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CleanedData.csv"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'
            return response

        except Exception as e:
            # Catch any other unexpected errors during the process
            messages.error(request, f"An unexpected error occurred during AI processing: {str(e)}")
            return redirect('dashboard') # Redirect to a safe page

    # --- GET request logic (when the page is first loaded) ---
    else:
        context = {
            'instruction_sets': instruction_sets,
            'default_instruction': default_instruction
        }
        return render(request, 'leads/process_digital_data.html', context)


@login_required
def process_physical_data_view(request):
    """
    Handles the page for processing physical documents (like images).
    Processes data using the Gemini Vision API, tracks token usage, and calculates cost.
    Creates a TransactionRecord for each successful processing event.
    """
    # This initial setup is the same
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    default_instruction = instruction_sets.filter(is_default=True).first()
    context = {
        'instruction_sets': instruction_sets,
        'default_instruction': default_instruction
    }

        # 2. THE CRITICAL CHECK: Does the user have ANY instructions at all?
    if not instruction_sets.exists():
        # If the user has zero instructions, send a helpful message and redirect them.
        messages.info(request, "Welcome! Please create your first AI instruction set to get started.")
        return redirect('manage_instructions') # Redirect to the page where they can create one.

    # 3. If they have instructions, find the default one for the form.
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method != 'POST':
        # If it's a GET request, just show the upload page
        return render(request, 'leads/process_physical_data.html', context)

    # --- Start of POST request logic ---
    
    # 1. Check User Quota
    profile = request.user.profile
    profile.check_and_reset_quota()
    if profile.cleans_this_month >= profile.monthly_quota:
        messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
        return redirect('dashboard')

    # 2. Get Instructions
    selected_instruction_id = request.POST.get('selected_instruction_id')
    selected_instruction = None
    if selected_instruction_id:
        selected_instruction = instruction_sets.filter(pk=selected_instruction_id).first()
    
    if not selected_instruction:
        selected_instruction = default_instruction

    if not selected_instruction:
        messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
        return redirect('manage_instructions')

    user_instructions = selected_instruction.instructions

    # 3. Process Uploaded Images
    uploaded_images = request.FILES.getlist('physical_images')
    if not uploaded_images:
        messages.error(request, "Please upload at least one image file.")
        return render(request, 'leads/process_physical_data.html', context)

    image_parts = []
    supported_image_types = ['image/jpeg', 'image/png', 'image/webp']
    for uploaded_image in uploaded_images:
        mime_type, _ = mimetypes.guess_type(uploaded_image.name)
        if mime_type not in supported_image_types:
            messages.warning(request, f"Skipping unsupported image type: {uploaded_image.name}.")
            continue
        image_parts.append({"mime_type": mime_type, "data": uploaded_image.read()})

    if not image_parts:
        messages.error(request, "No valid image files were provided for processing.")
        return render(request, 'leads/process_physical_data.html', context)

    # 4. Call AI and Handle Success/Failure
    try:
        # This function should be designed to return a tuple like:
        # (text_output, input_tokens, output_tokens) on success
        # (None, 0, 0) on failure, with the error logged internally
        extracted_data_raw_output, input_tokens, output_tokens = call_gemini_vision_api(image_parts, user_instructions)

        # --- THIS IS THE CRITICAL CHECK ---
        # If the function failed, the text output will be None (or you can check tokens)
        if extracted_data_raw_output is None:
            messages.error(request, "The AI processing failed. This could be due to a network issue, an invalid API key, or an issue with the content sent. Please try again.")
            return render(request, 'leads/process_physical_data.html', context)

        # If the AI returned an empty string, it's a warning, not a crash
        if not extracted_data_raw_output.strip():
            messages.warning(request, "AI returned an empty response. Please check your instructions and image content.")
            return render(request, 'leads/process_physical_data.html', context)
        
        # --- IF WE REACH HERE, THE AI CALL WAS SUCCESSFUL ---
        
        # 5. Calculate Cost and Update Profile
        model_used = 'gemini-1.5-flash' # The model used for vision
        cost = calculate_gemini_cost(model_used, input_tokens, output_tokens)
        cost = Decimal(str(cost)) # Ensure it's a Decimal

        profile.total_input_tokens += input_tokens
        profile.total_output_tokens += output_tokens
        profile.total_cost_usd += cost
        profile.cleans_this_month += 1
        profile.save()

        # 6. Create Transaction Record
        TransactionRecord.objects.create(
            user=request.user,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            transaction_type='physical',
        )

        # 7. Create and Return Excel File
        df = pd.read_csv(io.StringIO(extracted_data_raw_output), sep=',')
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ExtractedData')
        excel_buffer.seek(0)

        response = HttpResponse(excel_buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ExtractedImageData.xlsx"
        response['Content-Disposition'] = f'attachment; filename="{default_name}"'
        return response

    except pd.errors.ParserError as e:
        # Handle the case where the AI returns text that isn't valid CSV
        messages.error(request, f"AI returned data in an unreadable format. Please refine your instructions to be more specific about CSV output. Error: {str(e)}")
        return render(request, 'leads/process_physical_data.html', context)
    except Exception as e:
        # This is a catch-all for any other unexpected errors
        messages.error(request, f"An unexpected critical error occurred: {str(e)}")
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