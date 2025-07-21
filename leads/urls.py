# In leads/urls.py

from django.urls import path
# --- 1. IMPORT the built-in LogoutView ---
from django.contrib.auth.views import LogoutView
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('signup/', views.signup_view, name='signup'),
    path('signin/', views.signin_view, name='signin'),
    
    # --- 2. REPLACE your custom signout_view with the built-in LogoutView ---
    # This path will handle the logout and redirect to the homepage ('index').
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),
    
    # --- (Your old signout path is now removed) ---
    # path('signout/', views.signout_view, name='signout'),

    # Your existing paths for processing and instructions (these are all correct)
    path('process-digital-data/', views.process_digital_data_view, name='process_digital_data'),
    path('process-physical-data/', views.process_physical_data_view, name='process_physical_data'),
    path('manage_instructions/', views.manage_instructions, name='manage_instructions'),
    path('create_instruction/', views.create_instruction, name='instruction_form'),
    path('edit_instruction/<int:pk>/', views.edit_instruction, name='edit_instruction'),
    path('delete_instruction/<int:pk>/', views.delete_instruction, name='delete_instruction'),
    path('set_default_instruction/<int:pk>/', views.set_default_instruction, name='set_default_instruction'),
]