from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from main.models import ModelUser, Condition
from django.contrib import messages
from main.detect import upload_and_predict
#from main.ml.inference import get_recommendation
from skin_condition_analysis import settings
import os

def index(request):
    profile = ModelUser.objects.get(user=request.user)
    context ={'profile': profile}
    return render(request, "main/base.html", context)

def home(request):
    return render(request, "main/home.html")

def register(request):
    if request.method =="POST":
        username = request.POST['username'] 
        fname = request.POST['fname']
        lname = request.POST['lname'] 
        email = request.POST['email']       
        password = request.POST['password']
        pass2 = request.POST['pass2']
        phonenumber = request.POST['phonenumber']
        gender = request.POST['gender'] 
        modeluser_image = request.FILES.get('profile_picture') 
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try again")
            return redirect('register')
        
        if User.objects.filter(email=email):
            messages.error(request,"Email address is already registered!")
            return redirect('register')
    
        if password != pass2:
            messages.error(request, "Passwords did not match!")        


        if modeluser_image:
            # Construct the file path using MEDIA_ROOT
            file_path = os.path.join(settings.MEDIA_ROOT, 'profile_picture.jpg')

            # Open the file using the constructed file path
            with open(file_path, 'wb+') as destination:
                for chunk in modeluser_image.chunks():
                        destination.write(chunk)

        model_user = User.objects.create_user(username=username, email = email, password=password, first_name = fname, last_name = lname)
        modeluser= ModelUser(user = model_user, phone_number=phonenumber, gender=gender, modeluser_image = modeluser_image)  
        modeluser.save()                              
        return redirect('login')

    return render(request, "main/register.html")


def custom_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None:
                login(request, user)
                username = user.username
                return render(request, "main/home.html", {'username': username})

        else:
                messages.error(request, "Wrong credentials")
                return redirect('register')

    return render(request, "main/login.html")


def analysis(request):
    if request.method =="POST":
        condition_image = request.FILES.get('condition_image') 
          
        if condition_image:
            # Construct the file path using MEDIA_ROOT
            file_path = os.path.join(settings.MEDIA_ROOT, condition_image.name)

            # Open the file using the constructed file path
            with open(file_path, 'wb+') as destination:
                for chunk in condition_image.chunks():
                    destination.write(chunk)

            predicted_label = upload_and_predict(file_path)
            user = request.user

            #recommendation = gpt_recommendation(predicted_label)
            #recommendation = get_recommendation(predicted_label)
            #recommendation=recommendation

        condition=Condition(user = user, condition_image = condition_image, condition = predicted_label)                                
        condition.save()    
        return redirect('analysis')

    condition = None
    profile = None
    if request.user.is_authenticated:
        try:
            condition = Condition.objects.filter(user=request.user).last()
        except (Condition.DoesNotExist, ModelUser.DoesNotExist):
            condition = None
            profile = None
    else:
        condition = None
        profile = None

    context = {
        'condition': condition,
        'profile': profile,
    }

    return render(request, "main/analysis.html", context)

def signout(request):
    logout(request)
    messages.success(request, 'Logout successfully')
    return redirect('home')

