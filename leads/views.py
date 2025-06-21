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
    Sends images and text instructions to a Gemini Vision model and returns extracted data,
    along with input/output token counts.
    image_parts: A list of dicts, e.g., [{"mime_type": "image/jpeg", "data": b"..."}]
    user_instructions: Text instructions for Gemini.
    Returns: tuple (cleaned_text, total_input_tokens, total_output_tokens)
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)

    # CRITICAL: Use a multimodal model for vision tasks. 'gemini-1.5-pro-latest' supports images.
    model_name = 'models/gemini-2.0-flash' 
    model = genai.GenerativeModel(model_name)

    total_input_tokens = 0
    total_output_tokens = 0

    content_parts = []
    content_parts.extend(image_parts) # Add image data
    content_parts.append({"text": user_instructions}) # Add text instructions

    try:
        response = model.generate_content(
            content_parts,
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

        # Access token usage from usage_metadata
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_input_tokens = response.usage_metadata.prompt_token_count
            total_output_tokens = response.usage_metadata.candidates_token_count

        cleaned_text = response.text.strip()
        # Robust cleaning for vision model output, similar to text model
        if cleaned_text.startswith("```csv"):
            cleaned_text = cleaned_text[len("```csv"):].strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        
        # Split into lines for more granular cleaning and pipe-delimited table conversion
        output_lines = cleaned_text.splitlines()
        processed_lines = []
        for i, line in enumerate(output_lines):
            stripped_line = line.strip()
            
            # Skip markdown table separator lines (like |---|---|)
            if all(c == '-' or c == '|' or c.isspace() for c in stripped_line) and len(stripped_line) > 0:
                continue 

            # Attempt to detect and remove conversational intros (e.g., "Okay here's the information...")
            if i == 0 and not ',' in stripped_line and not stripped_line.startswith('|'):
                if "okay" in stripped_line.lower() or "here's the information" in stripped_line.lower():
                    continue

            # If it's a pipe-delimited line, convert it to comma-delimited CSV
            if stripped_line.startswith('|') and stripped_line.endswith('|'):
                parts = [p.strip() for p in stripped_line.strip('|').split('|')]
                valid_parts = [p for p in parts if p] # Remove empty parts from extra pipes
                if valid_parts:
                    processed_lines.append(','.join(valid_parts))
            else:
                # Assume it's already a CSV line or other text that should be included
                processed_lines.append(stripped_line)
        
        final_cleaned_output = '\n'.join(processed_lines)
        
        if not final_cleaned_output.strip():
            raise RuntimeError("AI response was cleaned, but resulted in empty data.")
        
        return final_cleaned_output, total_input_tokens, total_output_tokens

    except Exception as e:
        raise RuntimeError(f"Gemini Vision API call failed: {e}. Please ensure your API key is correct and valid. If using the Generative AI library, ensure it's properly configured.")


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

# @login_required
def dashboard_view(request):
    """
    Renders the user dashboard page, showing current quota and aggregated usage.
    Also handles date range filtering for user's transaction history.
    """
    profile, created = Profile.objects.get_or_create(user=request.user)
    profile.check_and_reset_quota() # Ensure quota is up-to-date for the current month

    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    default_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()
    
    current_quota = profile.monthly_quota - profile.cleans_this_month
    total_quota = profile.monthly_quota
    show_admin_link = profile.is_admin # Determine if admin link should be shown

    # --- Date Range Filtering Logic for User Dashboard ---
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')

    # Start with all transaction records for the current user
    filtered_transactions = TransactionRecord.objects.filter(user=request.user)

    if start_date_str:
        try:
            # Parse start_date and make it timezone-aware (start of the day)
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            start_datetime = timezone.make_aware(datetime.combine(start_date, datetime.min.time()))
            filtered_transactions = filtered_transactions.filter(timestamp__gte=start_datetime)
        except ValueError:
            messages.error(request, "Invalid start date format. Please use YYYY-MM-DD.")
            start_date_str = '' # Clear invalid date to prevent pre-filling bad input
    
    if end_date_str:
        try:
            # Parse end_date and make it timezone-aware (end of the day)
            # Add one day to include the entire end_date, then use < for exclusive range
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            end_datetime = timezone.make_aware(datetime.combine(end_date + timedelta(days=1), datetime.min.time())) 
            filtered_transactions = filtered_transactions.filter(timestamp__lt=end_datetime)
        except ValueError:
            messages.error(request, "Invalid end date format. Please use YYYY-MM-DD.")
            end_date_str = '' # Clear invalid date

    # Aggregate totals for the filtered range using Sum for Decimal fields
    range_totals = filtered_transactions.aggregate(
        sum_input_tokens=Sum('input_tokens', default=Decimal('0')), # Ensure default is Decimal
        sum_output_tokens=Sum('output_tokens', default=Decimal('0')), # Ensure default is Decimal
        sum_cost_usd=Sum('cost_usd', default=Decimal('0.00')) # Ensure default is Decimal
    )

    context = {
        'instruction_sets': instruction_sets,
        'default_instruction': default_instruction,
        'current_quota': current_quota,
        'total_quota': total_quota,
        'show_admin_link': show_admin_link,
        
        # All-time totals from Profile model
        'total_input_tokens': profile.total_input_tokens,
        'total_output_tokens': profile.total_output_tokens,
        'total_cost_usd': profile.total_cost_usd,

        # For date range results on dashboard
        'range_input_tokens': range_totals['sum_input_tokens'],
        'range_output_tokens': range_totals['sum_output_tokens'],
        'range_cost_usd': range_totals['sum_cost_usd'],
        'start_date': start_date_str, # Pass back to template for form input values
        'end_date': end_date_str,     # Pass back to template for form input values
    }
    return render(request, 'leads/dashboard.html', context)


@login_required
def process_digital_data_view(request):
    """
    Handles the page for processing digital files (like Excel/CSV) and pasted text.
    Processes data using the Gemini API, tracks token usage, and calculates cost.
    Creates a TransactionRecord for each successful processing event.
    """
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method == 'POST':
        profile = request.user.profile
        profile.check_and_reset_quota() # Ensure user's quota is up-to-date

        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
            return redirect('dashboard') 

        pasted_text = request.POST.get('pasted_text', '')
        uploaded_files = request.FILES.getlist('digital_files')

        selected_instruction_id = request.POST.get('selected_instruction_id')
        selected_instruction = None
        if selected_instruction_id:
            try:
                selected_instruction = get_object_or_404(InstructionSet, pk=selected_instruction_id, user=request.user)
            except Exception:
                messages.warning(request, "Selected instruction not found or does not belong to you. Attempting to use default.")
        
        if not selected_instruction:
            selected_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()

        if not selected_instruction:
            messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
            return redirect('manage_instructions')

        user_instructions = selected_instruction.instructions.strip()

        if not pasted_text and not uploaded_files:
            messages.error(request, "Please upload at least one file or paste some text data.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
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
                                    if max_columns_found > 1:
                                        break
                            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                                continue
                            except Exception as e:
                                print(f"DEBUG: Error trying encoding '{encoding_attempt}' and delimiter '{delimiter_attempt}' for '{uploaded_file.name}': {e}")
                                continue
                        if max_columns_found > 1:
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
                    if file_extension not in ['txt']:
                        df['_source_file'] = uploaded_file.name
                    combined_data.append(df)
                else:
                    messages.warning(request, f"File '{uploaded_file.name}' was empty or could not be read after parsing.")
            except Exception as e:
                messages.error(request, f"Failed to process '{uploaded_file.name}': {e}")
                continue

        if pasted_text:
            pasted_df = pd.DataFrame([{'raw_input_data': pasted_text, '_source_file': 'Pasted_Data'}])
            combined_data.append(pasted_df)

        if not combined_data:
            messages.error(request, "No valid data found after processing files and text.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)

        combined_input_df = pd.concat(combined_data, ignore_index=True)
        combined_input_df = combined_input_df.fillna('')

        gemini_prompt = format_gemini_prompt(combined_input_df, user_instructions)

        try:
            cleaned_csv_output_tuple = call_gemini_api(gemini_prompt)
            
            cleaned_csv_output = cleaned_csv_output_tuple[0]
            input_tokens = cleaned_csv_output_tuple[1]
            output_tokens = cleaned_csv_output_tuple[2]

            model_used = 'gemini-2.0-flash' 

            if not cleaned_csv_output.strip():
                messages.warning(request, "Gemini returned an empty response. Please check your instructions and input data.")
                context = {
                    'instruction_sets': instruction_sets,
                    'default_instruction': selected_instruction
                }
                return render(request, 'leads/process_digital_data.html', context)

            # --- Calculate cost and update user's profile totals ---
            cost = calculate_gemini_cost(model_used, input_tokens, output_tokens)
            
            # Ensure 'cost' is Decimal for addition to profile.total_cost_usd (safeguard)
            if not isinstance(cost, Decimal):
                cost = Decimal(str(cost)) 

            profile.total_input_tokens += input_tokens
            profile.total_output_tokens += output_tokens
            profile.total_cost_usd += cost
            profile.cleans_this_month += 1
            profile.save()

            # --- Create a TransactionRecord for this usage event ---
            TransactionRecord.objects.create(
                user=request.user,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                transaction_type='digital',
                # model_used=model_used, # Uncomment if you add 'model_used' field to TransactionRecord
            )

            response = HttpResponse(cleaned_csv_output, content_type='text/csv')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CleanedLeads.csv"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'

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
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method == 'POST':
        profile = request.user.profile
        profile.check_and_reset_quota()

        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
            return redirect('dashboard')

        uploaded_images = request.FILES.getlist('physical_images')
        
        selected_instruction_id = request.POST.get('selected_instruction_id')
        selected_instruction = None
        if selected_instruction_id:
            try:
                selected_instruction = get_object_or_404(InstructionSet, pk=selected_instruction_id, user=request.user)
            except Exception:
                messages.warning(request, "Selected instruction not found or does not belong to you. Attempting to use default.")
        
        if not selected_instruction:
            selected_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()

        if not selected_instruction:
            messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
            return redirect('manage_instructions')

        user_instructions = selected_instruction.instructions.strip()

        if not uploaded_images:
            messages.error(request, "Please upload at least one image file.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)

        image_parts = []
        supported_image_types = ['image/jpeg', 'image/png', 'image/webp'] 
        
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
            extracted_data_raw_output_tuple = call_gemini_vision_api(image_parts, user_instructions)

            extracted_data_raw_output = extracted_data_raw_output_tuple[0]
            input_tokens = extracted_data_raw_output_tuple[1]
            output_tokens = extracted_data_raw_output_tuple[2]
            
            model_used = 'gemini-2.0-flash' # The vision model used for image processing

            if not extracted_data_raw_output.strip():
                messages.warning(request, "AI returned an empty response for image processing. Please check your instructions and image content.")
                context = {
                    'instruction_sets': instruction_sets,
                    'default_instruction': selected_instruction
                }
                return render(request, 'leads/process_physical_data.html', context)

            # --- Calculate cost and update user's profile totals ---
            cost = calculate_gemini_cost(model_used, input_tokens, output_tokens)
            
            # Ensure 'cost' is Decimal for addition to profile.total_cost_usd (safeguard)
            if not isinstance(cost, Decimal):
                cost = Decimal(str(cost)) 

            profile.total_input_tokens += input_tokens
            profile.total_output_tokens += output_tokens
            profile.total_cost_usd += cost
            profile.cleans_this_month += 1
            profile.save()

            # --- Create a TransactionRecord for this usage event ---
            TransactionRecord.objects.create(
                user=request.user,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                transaction_type='physical',
                # model_used=model_used, # Uncomment if you add 'model_used' field to TransactionRecord
            )

            df = pd.read_csv(io.StringIO(extracted_data_raw_output), sep=',')

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='ExtractedData')
            excel_buffer.seek(0)

            response = HttpResponse(excel_buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ExtractedImageData.xlsx"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'

            return response

        except RuntimeError as e:
            messages.error(request, f"AI Processing Error: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_physical_data.html', context)
        except pd.errors.ParserError as e:
            messages.error(request, f"AI returned data in an unreadable CSV format. Please refine your instructions. (Error: {e}) Raw AI output might be: {extracted_data_raw_output[:200]}...")
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
    
    else:
        context = {
            'instruction_sets': instruction_sets,
            'default_instruction': default_instruction
        }
        return render(request, 'leads/process_physical_data.html', context)


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
    """
    Handles the page for processing digital files (like Excel/CSV) and pasted text.
    Processes data using the Gemini API, tracks token usage, and calculates cost.
    """
    # Fetch all instructions for the current user to populate the modal
    instruction_sets = InstructionSet.objects.filter(user=request.user).order_by('name')
    
    # Get the default instruction to pre-select it and display its name
    default_instruction = instruction_sets.filter(is_default=True).first()

    if request.method == 'POST':
        profile = request.user.profile
        # Ensure the user's quota is checked and reset if it's a new month
        profile.check_and_reset_quota()

        # Check if the user has exceeded their monthly data cleaning limit
        if profile.cleans_this_month >= profile.monthly_quota:
            messages.error(request, f"You have reached your monthly data cleaning limit ({profile.monthly_quota}). Please contact an administrator for an increase.")
            # Redirect to the dashboard or re-render the current page with the error
            return redirect('dashboard') 

        pasted_text = request.POST.get('pasted_text', '')
        # IMPORTANT: This matches 'name="digital_files"' from your HTML form
        uploaded_files = request.FILES.getlist('digital_files')

        # --- Instruction Selection Logic ---
        # Get the ID of the instruction set selected by the user
        selected_instruction_id = request.POST.get('selected_instruction_id')
        selected_instruction = None # Initialize to None

        if selected_instruction_id:
            try:
                # Attempt to retrieve the selected instruction set, ensuring it belongs to the current user
                selected_instruction = get_object_or_404(InstructionSet, pk=selected_instruction_id, user=request.user)
            except Exception:
                # If the selected instruction is not found or doesn't belong to the user,
                # log a warning and attempt to use the default instruction instead.
                messages.warning(request, "Selected instruction not found or does not belong to you. Attempting to use default.")
        
        # If no valid instruction was explicitly selected, or if the selected one failed,
        # try to find the user's default instruction set.
        if not selected_instruction:
            selected_instruction = InstructionSet.objects.filter(user=request.user, is_default=True).first()

        # If, after all attempts, no instruction set is found (meaning the user has none, or no default),
        # inform the user and redirect them to the instruction management page.
        if not selected_instruction:
            messages.error(request, "No AI instructions found. Please create one or set a default in 'Manage AI Instructions'.")
            return redirect('manage_instructions') # Redirect to manage instructions page

        # Extract the actual instructions text from the selected instruction set
        user_instructions = selected_instruction.instructions.strip()

        # Check if any data (pasted text or uploaded files) was provided
        if not pasted_text and not uploaded_files:
            messages.error(request, "Please upload at least one file or paste some text data.")
            # If no data, re-render the form, preserving the selected instruction context
            context = {
                'instruction_sets': instruction_sets, # All available instruction sets
                'default_instruction': selected_instruction # The instruction that was attempted to be used
            }
            return render(request, 'leads/process_digital_data.html', context)

        combined_data = [] # List to hold DataFrames from various sources

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            try:
                _, file_extension = os.path.splitext(uploaded_file.name)
                file_extension = file_extension.lower().lstrip('.')

                # Handle CSV files with robust encoding and delimiter detection
                if file_extension == 'csv':
                    best_df = pd.DataFrame()
                    raw_file_content_bytes = uploaded_file.read()

                    detected_encoding = None
                    try:
                        # Use chardet to detect file encoding for CSVs
                        result = chardet.detect(raw_file_content_bytes)
                        detected_encoding = result['encoding']
                        if detected_encoding:
                            detected_encoding = detected_encoding.lower()
                            # Treat 'ascii' as 'utf-8' for common compatibility
                            if detected_encoding == 'ascii':
                                detected_encoding = 'utf-8'
                    except Exception as e:
                        print(f"DEBUG: chardet failed for '{uploaded_file.name}': {e}")
                        detected_encoding = None

                    # Define a list of encodings to try, prioritizing detected one
                    encodings_to_try = []
                    if detected_encoding: encodings_to_try.append(detected_encoding)
                    if 'utf-8' not in encodings_to_try: encodings_to_try.append('utf-8')
                    if 'utf-16' not in encodings_to_try: encodings_to_try.append('utf-16')
                    if 'latin1' not in encodings_to_try: encodings_to_try.append('latin1')
                    if 'cp1252' not in encodings_to_try: encodings_to_try.append('cp1252')

                    # Define common delimiters to try
                    delimiters_to_try = [',', ';', '\t', '|']
                    max_columns_found = 0 # Track best parse attempt by number of columns

                    # Iterate through encodings and delimiters to find the best parse
                    for encoding_attempt in encodings_to_try:
                        for delimiter_attempt in delimiters_to_try:
                            try:
                                decoded_csv_data = raw_file_content_bytes.decode(encoding_attempt)
                                temp_df = pd.read_csv(
                                    io.StringIO(decoded_csv_data),
                                    sep=delimiter_attempt,
                                    on_bad_lines='skip', # Skip problematic lines
                                    header=0 # Assume first row is header
                                )

                                # Update best_df if a better parse (more columns) is found
                                if not temp_df.empty and temp_df.shape[1] > max_columns_found:
                                    best_df = temp_df
                                    max_columns_found = temp_df.shape[1]
                                    if max_columns_found > 1: # If already found multiple columns, likely a good parse
                                        break # Exit inner loop (delimiters)
                            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                                # Continue to next attempt on parsing errors
                                continue
                            except Exception as e:
                                # Catch any other unexpected errors during parsing attempt
                                print(f"DEBUG: Error trying encoding '{encoding_attempt}' and delimiter '{delimiter_attempt}' for '{uploaded_file.name}': {e}")
                                continue
                        if max_columns_found > 1: # If a good parse was found, exit outer loop (encodings)
                            break
                    
                    # If after all attempts, the CSV still couldn't be parsed well
                    if best_df.empty or max_columns_found <= 1:
                        messages.error(request, f"Failed to parse CSV columns from '{uploaded_file.name}' after trying multiple encodings and delimiters. Please check the file's actual format and contents.")
                        continue # Skip to the next uploaded file
                    else:
                        df = best_df # Use the best parsed DataFrame

                # Handle Excel files
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(uploaded_file)
                
                # Handle plain text files
                elif file_extension == 'txt':
                    try:
                        text_content = uploaded_file.read().decode('utf-8')
                        # For text files, put content into a 'raw_input_data' column
                        df = pd.DataFrame([{'raw_input_data': text_content, '_source_file': uploaded_file.name}])
                    except Exception as e:
                        messages.error(request, f"Failed to read text file '{uploaded_file.name}': {e}")
                        continue # Skip to the next uploaded file
                
                # Handle unsupported file types
                else:
                    messages.warning(request, f"Skipping unsupported file type: {uploaded_file.name}")
                    continue

                # If the DataFrame is not empty after parsing, add it to the combined list
                if not df.empty:
                    if file_extension not in ['txt']: # Only add _source_file if not already added by text parsing
                        df['_source_file'] = uploaded_file.name
                    combined_data.append(df)
                else:
                    messages.warning(request, f"File '{uploaded_file.name}' was empty or could not be read after parsing.")
            except Exception as e:
                # Catch any general errors during file processing
                messages.error(request, f"Failed to process '{uploaded_file.name}': {e}")
                continue

        # Process pasted text if it exists
        if pasted_text:
            # Create a DataFrame for pasted text, similar to how text files are handled
            pasted_df = pd.DataFrame([{'raw_input_data': pasted_text, '_source_file': 'Pasted_Data'}])
            combined_data.append(pasted_df)

        # If no valid data was collected from files or pasted text, show an error
        if not combined_data:
            messages.error(request, "No valid data found after processing files and text.")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)

        # Concatenate all collected DataFrames into a single input DataFrame for Gemini
        combined_input_df = pd.concat(combined_data, ignore_index=True)
        combined_input_df = combined_input_df.fillna('') # Replace NaN with empty strings

        # Format the combined DataFrame and user instructions into a prompt for Gemini
        gemini_prompt = format_gemini_prompt(combined_input_df, user_instructions)

        try:
            # Call the updated Gemini API function. It now returns a tuple:
            # (cleaned_text_output, input_tokens_count, output_tokens_count)
            cleaned_csv_output_tuple = call_gemini_api(gemini_prompt)
            
            # Unpack the tuple to get the individual components
            cleaned_csv_output = cleaned_csv_output_tuple[0] # The actual text output from Gemini
            input_tokens = cleaned_csv_output_tuple[1]       # Number of input tokens used
            output_tokens = cleaned_csv_output_tuple[2]      # Number of output tokens generated

            # Specify the model used for cost calculation
            model_used = 'gemini-2.0-flash' 

            # Check if Gemini returned an empty response after cleaning (e.g., if instructions led to no output)
            if not cleaned_csv_output.strip():
                messages.warning(request, "Gemini returned an empty response. Please check your instructions and input data.")
                context = {
                    'instruction_sets': instruction_sets,
                    'default_instruction': selected_instruction
                }
                return render(request, 'leads/process_digital_data.html', context)

            # --- Calculate cost and update user's profile ---
            cost = calculate_gemini_cost(model_used, input_tokens, output_tokens)
            
            # Update the user's profile with the new token usage and calculated cost
            profile.total_input_tokens += input_tokens
            profile.total_output_tokens += output_tokens
            profile.total_cost_usd += cost
            profile.cleans_this_month += 1 # Increment the monthly cleaning count
            profile.save() # Save the updated profile to the database

            # Prepare the cleaned CSV output for download
            response = HttpResponse(cleaned_csv_output, content_type='text/csv')
            default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CleanedLeads.csv"
            response['Content-Disposition'] = f'attachment; filename="{default_name}"'

            return response

        except RuntimeError as e:
            # Catch specific errors from the AI API call
            messages.error(request, f"AI Processing Error: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)
        except Exception as e:
            # Catch any other unexpected errors during the overall data processing
            messages.error(request, f"An unexpected error occurred during data processing: {e}")
            context = {
                'instruction_sets': instruction_sets,
                'default_instruction': selected_instruction
            }
            return render(request, 'leads/process_digital_data.html', context)
    
    # --- This block handles the GET request for the page ---
    # When the page is first loaded (not a POST request), render the form normally.
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