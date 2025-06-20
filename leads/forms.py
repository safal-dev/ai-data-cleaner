# leads/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Profile

class UserRegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove help text for the username field
        if 'username' in self.fields:
            self.fields['username'].help_text = ''

        # You might also want to set password help_text to '' if it exists,
        # but the main verbose messages come from validators in settings.py
        if 'password' in self.fields:
            self.fields['password'].help_text = ''
        if 'password2' in self.fields:
            self.fields['password2'].help_text = ''

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
        return user

class UserLoginForm(AuthenticationForm):
    username = forms.CharField(label='Username', max_length=150)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)