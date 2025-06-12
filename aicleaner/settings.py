"""
Django settings for aicleaner project.
Refactored for production using django-environ.
"""
import os
import environ ### Import django-environ
from pathlib import Path

# Initialize django-environ
env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Take environment variables from .env file
# This assumes your .env file is in the root directory (with manage.py)
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))


# --- CORE SETTINGS READ FROM .ENV ---

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY') ### Reading from .env

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG') ### Reading from .env and casting to boolean

# ALLOWED_HOSTS is now a comma-separated list in .env
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS') ### Reading from .env

# --- YOUR CUSTOM APP SETTINGS ---
GEMINI_API_KEY = env('GEMINI_API_KEY') ### Reading from .env


# --- APPLICATION DEFINITION ---

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'leads',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    # Whitenoise is great for serving static files in production easily
    # 'whitenoise.middleware.WhiteNoiseMiddleware', ### Optional but recommended
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'aicleaner.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], ### Using pathlib for cleaner paths
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'aicleaner.wsgi.application'


# --- DATABASE ---
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
# This single line reads the DATABASE_URL from .env and configures the database.
# It works for SQLite, PostgreSQL, and MySQL.
DATABASES = {
    'default': env.db(), ### The magic of django-environ!
}


# --- PASSWORD VALIDATION ---
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators
# ### Re-enabled these for security. It's very important.
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]


# --- INTERNATIONALIZATION ---
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# --- STATIC AND MEDIA FILES ---
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
# ### THIS IS CRITICAL FOR PRODUCTION ###
# This is the directory where `collectstatic` will gather all static files.
# Your web server (like Nginx) will serve files from this folder.
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


# --- AUTHENTICATION & REDIRECTS ---
LOGIN_URL = 'signin'
LOGIN_REDIRECT_URL = 'index'
LOGOUT_REDIRECT_URL = 'index'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'