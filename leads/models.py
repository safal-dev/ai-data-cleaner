# leads/models.py

from django.db import models
from django.contrib.auth.models import User # Import User for ForeignKey
from django.db.models.signals import post_save # For creating profile automatically
from django.dispatch import receiver # For connecting signals
from datetime import date # Import date for the monthly reset logic
from decimal import Decimal # Import Decimal for precise currency calculations
from django.contrib.auth import get_user_model # More robust way to get the User model

# Get the custom User model if you're using one, otherwise defaults to auth.User
User = get_user_model()

# --- Profile Model ---
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_admin = models.BooleanField(default=False)
    monthly_quota = models.IntegerField(default=10) # Default 10 cleans per month
    cleans_this_month = models.IntegerField(default=0)
    last_quota_reset = models.DateField(auto_now_add=True) # Tracks when quota was last reset

    # New fields for token and cost tracking
    total_input_tokens = models.BigIntegerField(default=0)
    total_output_tokens = models.BigIntegerField(default=0)
    # Using DecimalField for currency to avoid floating-point inaccuracies
    total_cost_usd = models.DecimalField(max_digits=12, decimal_places=6, default=Decimal('0.00')) 
    # max_digits increased to 12 to accommodate larger potential costs,
    # decimal_places to 6 for high precision as per Gemini pricing.

    def __str__(self):
        return f"{self.user.username} Profile"

    # This method can be called to check and reset quota
    def check_and_reset_quota(self):
        today = date.today()
        # If the current month is different from the month of the last reset
        if self.last_quota_reset.month != today.month or self.last_quota_reset.year != today.year:
            self.cleans_this_month = 0
            # --- Reset token and cost usage too ---
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = Decimal('0.00') # Ensure reset to Decimal zero
            # -------------------------------------
            self.last_quota_reset = today
            self.save() # Save the updated profile

# Signal to create a Profile automatically when a new User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    # Ensure profile exists before saving (might be an issue if profile wasn't created initially)
    if hasattr(instance, 'profile'):
        instance.profile.save()


# --- InstructionSet Model ---
class InstructionSet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='instruction_sets')
    name = models.CharField(max_length=255, unique=False)
    instructions = models.TextField()
    is_default = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # Ensures that a user cannot have two instruction sets with the same name
        unique_together = ('user', 'name')
        # Orders instruction sets by their last update time, newest first
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.name} by {self.user.username}"

    def save(self, *args, **kwargs):
        # Logic to ensure only one instruction set is default per user
        if self.is_default:
            # If this instruction set is being set as default,
            # unset default for all other instruction sets for this user
            InstructionSet.objects.filter(user=self.user, is_default=True)\
                                 .exclude(pk=self.pk)\
                                 .update(is_default=False)
        super().save(*args, **kwargs) # Call the "real" save() method

    def delete(self, *args, **kwargs):
        # If the deleted instruction was the default, try to assign a new default
        if self.is_default:
            # Find another instruction set for the same user, excluding the one being deleted
            other_instruction = InstructionSet.objects.filter(user=self.user)\
                                                      .exclude(pk=self.pk)\
                                                      .first()
            if other_instruction:
                other_instruction.is_default = True
                other_instruction.save()
        super().delete(*args, **kwargs) # Call the "real" delete() method


# --- TransactionRecord Model (NEW) ---
class TransactionRecord(models.Model):
    """
    Stores individual transaction details for AI data processing.
    Each record represents one API call and its associated token usage and cost.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    timestamp = models.DateTimeField(auto_now_add=True) # Automatically set when created
    
    # Using DecimalField for token counts for consistency with cost and potential large numbers
    input_tokens = models.DecimalField(max_digits=15, decimal_places=0, default=Decimal('0'))
    output_tokens = models.DecimalField(max_digits=15, decimal_places=0, default=Decimal('0'))
    
    # Cost in USD, highly precise
    cost_usd = models.DecimalField(max_digits=12, decimal_places=6, default=Decimal('0.00'))
    
    transaction_type = models.CharField(
        max_length=50,
        choices=[
            ('digital', 'Digital Data Processing'),
            ('physical', 'Physical Data Processing'),
        ],
        help_text="Type of data processed (digital or physical)."
    )
    # You could add 'model_used = models.CharField(max_length=100, blank=True)' if you want to store model name per transaction

    class Meta:
        ordering = ['-timestamp'] # Default ordering: newest transactions first
        verbose_name = "Transaction Record"
        verbose_name_plural = "Transaction Records"

    def __str__(self):
        return f"Transaction for {self.user.username} on {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

class TransactionRecord(models.Model):
    # ... your existing fields (user, timestamp, input_tokens, etc.) ...

    # --- ADD THIS NEW FIELD ---
    processed_file = models.FileField(
        upload_to='processed_files/%Y/%m/%d/', # Organizes files by date
        null=True,
        blank=True,
        help_text="The final processed file available for download."
    )

    # You can also add a field for the original filename for clarity
    original_filename = models.CharField(max_length=255, blank=True, null=True)