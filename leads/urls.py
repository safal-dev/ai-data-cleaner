# leads/urls.py (YOUR APP'S urls.py)

from django.urls import path
from . import views # Correct relative import

# Remove these unnecessary imports from here. They belong in the project's urls.py
# from django.urls import path, include
# from django.contrib import admin

urlpatterns = [
    # REMOVE THIS LINE: path('admin/', admin.site.urls),
    # REMOVE THIS LINE: path('', include('leads.urls')), # This caused the RecursionError

    path('', views.index, name='index'), # This will now be your landing page
    path('dashboard/', views.dashboard_view, name='dashboard'), # New path for the actual processing forms
    path('signup/', views.signup_view, name='signup'),
    path('signin/', views.signin_view, name='signin'),
    path('signout/', views.signout_view, name='signout'),

    # Your existing paths for processing and instructions
    path('process-digital-data/', views.process_digital_data_view, name='process_digital_data'),
    path('process-physical-data/', views.process_physical_data_view, name='process_physical_data'),
    path('manage_instructions/', views.manage_instructions, name='manage_instructions'),
    path('create_instruction/', views.create_instruction, name='instruction_form'),
    path('edit_instruction/<int:pk>/', views.edit_instruction, name='edit_instruction'),
    path('delete_instruction/<int:pk>/', views.delete_instruction, name='delete_instruction'),
    path('set_default_instruction/<int:pk>/', views.set_default_instruction, name='set_default_instruction'),

    # REMOVE THESE ADMIN PATHS from here. They are handled by the project's urls.py
    # path('admin/users/', views.admin_user_management, name='admin_user_management'),
    # path('admin/users/edit/<int:pk>/', views.admin_edit_user, name='admin_edit_user'),
]