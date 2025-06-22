# leads/admin.py

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from django.urls import path, reverse
from django.template.response import TemplateResponse
from django import forms
from django.db.models import Sum
from decimal import Decimal
from datetime import datetime, timedelta
from django.utils import timezone
from django.utils.html import format_html
from django.shortcuts import redirect

from .models import InstructionSet, Profile, TransactionRecord 

# --- 1. Admin for InstructionSet ---

@admin.register(InstructionSet)
class InstructionSetAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_default', 'updated_at')
    list_filter = ('is_default',)
    search_fields = ('name', 'instructions')
    actions = ['set_as_default']

    def set_as_default(self, request, queryset):
        if queryset.count() > 1:
            self.message_user(request, "Please select only one instruction set to make default.", level='error')
            return

        selected_instruction_set = queryset.first()
        if not selected_instruction_set.is_default:
            InstructionSet.objects.update(is_default=False)
            
            selected_instruction_set.is_default = True
            selected_instruction_set.save()
            self.message_user(request, f"'{selected_instruction_set.name}' is now the default instruction set.")
        else:
            self.message_user(request, f"'{selected_instruction_set.name}' is already the default instruction set.")

    set_as_default.short_description = "Set selected instruction set as default"


# --- 2. Custom Admin for User and Profile ---

class ProfileInline(admin.StackedInline):
    """
    Inline for the Profile model to display it on the User admin page.
    """
    model = Profile
    can_delete = False
    verbose_name_plural = 'profile'
    readonly_fields = ('total_input_tokens', 'total_output_tokens', 'total_cost_usd', 'date_range_report_link')
    fields = ('is_admin', 'monthly_quota', 'cleans_this_month',
              'total_input_tokens', 'total_output_tokens', 'total_cost_usd',
              'date_range_report_link')

    @admin.display(description='View Detailed Usage for User')
    def date_range_report_link(self, obj):
        if obj and obj.user_id:
            # CRITICAL FIX: Ensure this reverse uses the full name that will be defined below
            url = reverse('admin:leads_transactionrecord_date_range_report') + f"?user={obj.user_id}"
            return format_html('<a href="{}" target="_blank" class="button">Open User Usage Report</a>', url)
        return "User not linked to profile"


class UserAdmin(BaseUserAdmin):
    """
    Custom UserAdmin class that integrates the ProfileInline.
    Also displays summary Profile fields directly in the User list view.
    """
    inlines = (ProfileInline,)
    list_display = (
        'username', 'email', 'first_name', 'last_name', 'is_staff',
        'get_monthly_quota', 'get_cleans_this_month', 'get_total_input_tokens',
        'get_total_output_tokens', 'get_total_cost_usd'
    )
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups', 'profile__is_admin')

    @admin.display(description='Monthly Quota')
    def get_monthly_quota(self, obj):
        # THE FIX: Check if the 'profile' attribute exists before accessing it.
        if hasattr(obj, 'profile'):
            return obj.profile.monthly_quota
        return "N/A"

    @admin.display(description='Cleans This Month')
    def get_cleans_this_month(self, obj):
        if hasattr(obj, 'profile'):
            return obj.profile.cleans_this_month
        return "N/A"

    @admin.display(description='Total Input Tokens (All-Time)')
    def get_total_input_tokens(self, obj):
        if hasattr(obj, 'profile'):
            return obj.profile.total_input_tokens
        return "N/A"

    @admin.display(description='Total Output Tokens (All-Time)')
    def get_total_output_tokens(self, obj):
        if hasattr(obj, 'profile'):
            return obj.profile.total_output_tokens
        return "N/A"

    @admin.display(description='Total Cost (USD - All-Time)')
    def get_total_cost_usd(self, obj):
        if hasattr(obj, 'profile'):
            return f"${obj.profile.total_cost_usd:.6f}"
        return "N/A"


# Unregister Django's default UserAdmin and register our custom one
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


# --- 3. Custom Admin for TransactionRecord (with Date Range Report View) ---

class DateRangeForm(forms.Form):
    """
    A simple Django form for selecting a date range and optionally filtering by user.
    Used in the custom admin report view.
    """
    start_date = forms.DateField(
        required=False,
        label="Start Date",
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'vDateField'})
    )
    end_date = forms.DateField(
        required=False,
        label="End Date",
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'vDateField'})
    )
    user = forms.ModelChoiceField(
        queryset=User.objects.all().order_by('username'),
        required=False,
        label="Filter by User",
        empty_label="-- All Users --"
    )


@admin.register(TransactionRecord)
class TransactionRecordAdmin(admin.ModelAdmin):
    """
    Admin configuration for the TransactionRecord model.
    Includes a custom report view accessible via a URL and an admin action.
    """
    list_display = (
        'user', 'timestamp', 'input_tokens', 'output_tokens', 'cost_usd', 'transaction_type'
    )
    list_filter = ('user', 'transaction_type', 'timestamp')
    search_fields = ('user__username',)


    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            # CRITICAL FIX: Change 'name' to the full convention
            path('date-range-report/', self.admin_site.admin_view(self.date_range_report_view), name='transactionrecord_date_range_report'),
        ]
        return custom_urls + urls


    def date_range_report_view(self, request):
        context = self.admin_site.each_context(request)
        
        initial_form_data = {}
        if 'user' in request.GET:
            try:
                initial_form_data['user'] = User.objects.get(pk=request.GET['user'])
            except User.DoesNotExist:
                pass
        
        form = DateRangeForm(request.GET or initial_form_data)
        
        sum_input_tokens = Decimal('0.00')
        sum_output_tokens = Decimal('0.00')
        sum_cost_usd = Decimal('0.00')
        
        transactions = TransactionRecord.objects.all()

        if form.is_valid():
            start_date = form.cleaned_data.get('start_date')
            end_date = form.cleaned_data.get('end_date')
            user_filter = form.cleaned_data.get('user')

            if user_filter:
                transactions = transactions.filter(user=user_filter)

            if start_date:
                start_datetime = timezone.make_aware(datetime.combine(start_date, datetime.min.time()))
                transactions = transactions.filter(timestamp__gte=start_datetime)
            
            if end_date:
                end_datetime = timezone.make_aware(datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
                transactions = transactions.filter(timestamp__lt=end_datetime)

            aggregates = transactions.aggregate(
                total_input=Sum('input_tokens', default=Decimal('0.00')),
                total_output=Sum('output_tokens', default=Decimal('0.00')),
                total_cost=Sum('cost_usd', default=Decimal('0.00'))
            )
            sum_input_tokens = aggregates['total_input']
            sum_output_tokens = aggregates['total_output']
            sum_cost_usd = aggregates['total_cost']

        context.update({
            'title': 'AI Usage Report by Date Range',
            'form': form,
            'sum_input_tokens': sum_input_tokens,
            'sum_output_tokens': sum_output_tokens,
            'sum_cost_usd': sum_cost_usd,
            'media': self.media,
            'has_permission': True
        })
        return TemplateResponse(request, 'admin/leads/transactionrecord/date_range_report.html', context)

    def view_date_range_report_global(self, request, queryset):
        # CRITICAL FIX: Ensure this redirect uses the full name defined above
        return redirect('admin:transactionrecord_date_range_report')
    view_date_range_report_global.short_description = "View Global Usage Report by Date Range"

    actions = [view_date_range_report_global]
