import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Create your views here.
token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
model_name = 'recogna-nlp/ptt5-base-summ'

tokenizer = T5Tokenizer.from_pretrained(token_name)
model_pt = T5ForConditionalGeneration.from_pretrained(model_name)


def summarizer(request):   
    return render(request, 'summarizer_app/text-summarizer.html', {'summary': ""})


def summary_result(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        min_size = int(request.POST.get('result-size'))
        
        inputs = tokenizer.encode(text,
                                  max_length=4096,
                                  truncation=True,
                                  return_tensors='pt'
                                  )
     
        summary_ids = model_pt.generate(inputs, max_length=640,
                                        min_length=min_size, num_beams=5,
                                        no_repeat_ngram_size=3, 
                                        early_stopping=True
                                        )
       
        summary = tokenizer.decode(summary_ids[0])
        
        response_data = {
            'message': 'Success',
            'summary': summary
        }

        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)