from django.conf.urls import include, url
from .views import KYC

urlpatterns = [
    url(r'api/v1/kyc/', KYC, name="kyc_documents"),

]
