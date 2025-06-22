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
# This part is correct and unchanged.
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
            # This logic can be simplified to avoid a race condition.
            # It sets all others to False first.
            queryset.model.objects.filter(user=request.user).update(is_default=False)
            selected_instruction_set.is_default = True
            selected_instruction_set.save()
            self.message_user(request, f"'{selected_instruction_set.name}' is now the default instruction set.")
        else:
            self.message_user(request, f"'{selected_instruction_set.name}' is already the default instruction set.")

    set_as_default.short_description = "Set selected instruction set as default"


# --- 2. Custom Admin for User and Profile ---
# This part is correct and unchanged.
class ProfileInline(admin.StackedInline):
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
            url = reverse('admin:leads_transactionrecord_date_range_report') + f"?user={obj.user_id}"
            return format_html('<a href="{}" target="_blank" class="button">Open User Usage Report</a>', url)
        return "User not linked to profile"

class UserAdmin(BaseUserAdmin):
    inlines = (ProfileInline,)
    list_display = (
        'username', 'email', 'first_name', 'last_name', 'is_staff',
        'get_monthly_quota', 'get_cleans_this_month', 'get_total_input_tokens',
        'get_total_output_tokens', 'get_total_cost_usd'
    )
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups', 'profile__is_admin')

    @admin.display(description='Monthly Quota')
    def get_monthly_quota(self, obj): return obj.profile.monthly_quota
    @admin.display(description='Cleans This Month')
    def get_cleans_this_month(self, obj): return obj.profile.cleans_this_month
    @admin.display(description='Total Input Tokens (All-Time)')
    def get_total_input_tokens(self, obj): return obj.profile.total_input_tokens
    @admin.display(description='Total Output Tokens (All-Time)')
    def get_total_output_tokens(self, obj): return obj.profile.total_output_tokens
    @admin.display(description='Total Cost (USD - All-Time)')
    def get_total_cost_usd(self, obj): return f"${obj.profile.total_cost_usd:.6f}"

admin.site.unregister(User)
admin.site.register(User, UserAdmin)


# --- 3. Custom Admin for TransactionRecord (with Corrected Report View) ---
# This part is correct and unchanged.
class DateRangeForm(forms.Form):
    start_date = forms.DateField(required=False, label="Start Date", widget=forms.DateInput(attrs={'type': 'date', 'class': 'vDateField'}))
    end_date = forms.DateField(required=False, label="End Date", widget=forms.DateInput(attrs={'type': 'date', 'class': 'vDateField'}))
    user = forms.ModelChoiceField(queryset=User.objects.all().order_by('username'), required=False, label="Filter by User", empty_label="-- All Users --")


@admin.register(TransactionRecord)
class TransactionRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'timestamp', 'input_tokens', 'output_tokens', 'cost_usd', 'transaction_type')
    list_filter = ('user', 'transaction_type', 'timestamp')
    search_fields = ('user__username',)

    # The get_urls method is correct and unchanged.
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('date-range-report/', self.admin_site.admin_view(self.date_range_report_view), name='transactionrecord_date_range_report'),
        ]
        return custom_urls + urls

    # --- THIS IS THE CORRECTED AND REFACTORED REPORT VIEW ---
    def date_range_report_view(self, request):
        context = self.admin_site.each_context(request)
        
        # Initialize the form with data from the request's GET parameters
        form = DateRangeForm(request.GET or None)
        
        # Start with a base queryset of all transactions
        queryset = TransactionRecord.objects.all()

        # Only try to filter IF the form is submitted AND valid.
        if form.is_valid():
            start_date = form.cleaned_data.get('start_date')
            end_date = form.cleaned_data.get('end_date')
            user_filter = form.cleaned_data.get('user')

            # The filtering logic is *inside* this block
            if user_filter:
                queryset = queryset.filter(user=user_filter)
            if start_date:
                start_datetime = timezone.make_aware(datetime.combine(start_date, datetime.min.time()))
                queryset = queryset.filter(timestamp__gte=start_datetime)
            if end_date:
                end_datetime = timezone.make_aware(datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
                queryset = queryset.filter(timestamp__lt=end_datetime)
        
        # The aggregation now runs OUTSIDE the 'if' block.
        # It runs on the full queryset on initial load, and on the filtered queryset after submission.
        aggregates = queryset.aggregate(
            total_input=Sum('input_tokens', default=Decimal('0')),
            total_output=Sum('output_tokens', default=Decimal('0')),
            total_cost=Sum('cost_usd', default=Decimal('0.00'))
        )

        # The context is updated with the CORRECT, calculated values from 'aggregates'.
        context.update({
            'title': 'AI Usage Report by Date Range',
            'form': form,
            'transactions': queryset.order_by('-timestamp'), # Pass the list of transactions
            'sum_input_tokens': aggregates['total_input'],   # Use the calculated value
            'sum_output_tokens': aggregates['total_output'], # Use the calculated value
            'sum_cost_usd': aggregates['total_cost'],      # Use the calculated value
            'media': self.media,
            'has_permission': True
        })
        
        return TemplateResponse(request, 'admin/leads/transactionrecord/date_range_report.html', context)

    # The admin action is correct and unchanged.
    def view_date_range_report_global(self, request, queryset):
        return redirect('admin:leads_transactionrecord_date_range_report')
    view_date_range_report_global.short_description = "View Global Usage Report by Date Range"

    actions = ['view_date_range_report_global']