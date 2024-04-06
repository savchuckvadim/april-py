from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


class TestView(APIView):
    def get(self, request, *args, **kwargs):
        phrase = request.GET.get('phrase', 'В космосе существует множество галактик и созвездий')
        input_ids = tokenizer.encode(phrase, return_tensors="pt")
        max_length = 50
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=1,  # Температура генерации, измените для разных стилей генерации
            do_sample=True,  # Включить сэмплирование; выключить для использования жадного декодирования
            top_k=50,  # Использовать только top_k токенов на каждом шаге
            top_p=0.95,  # Использовать только вероятные токены с общей вероятностью top_p
            num_return_sequences=1  # Количество возвращаемых последовательностей
        )
        # prompt = f"{phrase}"
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return JsonResponse({"message": generated_text})

        # Возвращаем найденный ответ в ответе
       
      

class TrainView(APIView):
    def get(self, request, *args, **kwargs):
        phrase = request.GET.get('phrase', 'Someone')
        training_args = TrainingArguments(
            output_dir="./results",          # выходной каталог для сохранения модели
            num_train_epochs=1,              # общее количество эпох обучения
            per_device_train_batch_size=4,   # размер батча для обучения
            per_device_eval_batch_size=4,    # размер батча для оценки
            warmup_steps=500,                # количество шагов для разогрева
            weight_decay=0.01,               # вес уменьшения
            logging_dir="./logs",            # каталог для сохранения логов
        )

        trainer = Trainer(
          model=model,                         # модель для дообучения
          args=training_args,                  # аргументы обучения
        #   train_dataset=train_dataset,         # набор данных для обучения
        #   eval_dataset=eval_dataset            # набор данных для оценки
        )

# Начать дообучение
        trainer.train()
  
        

        return JsonResponse({"message": 'output_text'})

        # Возвращаем найденный ответ в ответе
       
      
        