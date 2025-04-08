from django.db import models
from django.contrib.auth.models import User


class ModelUser(models.Model):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    GENDER_CHOICES =(
        ('male','Male'),
        ('female','Female')
    )

    modeluser_id = models.AutoField(primary_key= True)
    modeluser_image = models.ImageField(null=True, blank=True)
    phone_number = models.IntegerField(help_text='0712345678 or +254712345678', null=True, blank=True) 
    gender = models.CharField(max_length=10, null=True, blank=True, choices = GENDER_CHOICES)

         
    def __str__(self):
        return str(self.modeluser_id)
    class Meta:
        db_table='modelUser'


    @property #help us access this as an attribute rather than as a model
    def modeluser_imageURL(self):
        try: 
            url = self.modeluser_image.url
        except:
            url = ''
        return url
    

class Condition(models.Model): 
    user= models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    condition_id = models.AutoField(primary_key=True)
    condition = models.CharField(max_length=10, null=True, blank=True)
    condition_image = models.ImageField(null=True, blank=True)
    #recommendation = models.CharField(null=True, blank=True, max_length=5000)
    def __str__(self):
        return str(self.condition_id)
    class Meta:
        db_table='condition'


    @property #help us access this as an attribute rather than as a model
    def condition_imageURL(self):
        try: 
            url = self.condition_image.url
        except:
            url = ''
        return url
    






    







