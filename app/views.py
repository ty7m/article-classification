from django.http import HttpResponse
from django.template import loader
from .python import classification


def input(request):
    template = loader.get_template("app/input.html")
    return HttpResponse(template.render({}, request))


def output(request):
    categoly_list = ["エンタメ", " スポーツ", "おもしろ", "国内",
                     "海外", "コラム", "IT・科学", "グルメ"]
    input_url = request.POST["url"]
    pre = classification.main(input_url)
    categoly = categoly_list[pre]

    template = loader.get_template("app/output.html")
    context = {
            "url": input_url,
            "categoly": categoly,
            }
    return HttpResponse(template.render(context, request))
