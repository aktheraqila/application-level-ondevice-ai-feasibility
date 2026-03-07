from transformers import pipeline
import time
import statistics

print("Loading t5-small model...")

summarizer = pipeline("summarization", model="t5-small")

text = """

The US has the largest economy across the globe. However, it has been experiencing challenges that are associated with recession following the 2008 economic crisis. Unemployment constitutes one of the biggest problems in the US.
As Rothstein and Valletta (2014) confirm, after the recession, unemployment has been one of the major challenges that are facing US policy developers. Without adequate employment opportunities, the confidence of consumers decreases. As the paper confirms, to address the problems of inflation, unemployment, and recession, the US government has adopted various measures to revamp the economic growth.
The US has been exploring ‘quantitative easing’ measures. Through the measures, the government has increased its expenditure in an effort to boost its economic growth. It also continues to explore various mechanisms of rescuing insurers and banks that were negatively affected by the economic crisis. The main goal is to restore investor confidence and public trust. For instance, FOMC resorted to purchasing administration securities to reduce interest charges whilst promoting savings. A growing investment amplifies self-assurance and production capacity.
Therefore, through macroeconomic policies, the government of the United States is currently attempting to revive and ensure economic success following the recession. In 2014, good news to the Americans is that the nation has experienced continuous economic growth over the last three years since the recession.Amid efforts by the government of United States to address core problems such as unemployment, recession, and inflation, challenges remain. For instance, since the recession, the nation has realized an average economic growth of about 2.25% (OECD Economic Outlook, 2014).The US is also experiencing budget problems, which also impede its expected high growth rates. The government has adopted a fiscal cliff deal of about $1.1 trillion with the objective of reducing yearly deficits. Judging from the impacts of the deal, it has done a little to achieve its noble objective. Some of the deficits that are currently being experienced are related to nominal effects of its weak economy.
Despite the challenges that the US economy has witnessed since the recession, an incredible success has been achieved through the adoption of requisite fiscal policies. For instance, since 2012, unemployment has been reducing steadily.By January 2013, more than 155, 000 jobs were created. Creation of jobs at the rate of 300, 000 new ones every year is necessary to reduce unemployment levels in the right speed (Sivy, 2014). In the context of inflation, policymakers enjoy when the yearly inflation falls in the range of 0 and 2%.Over 2013 and 2014, consumer prices have only increased at the rate of about 1.8% per year. In 2013, through the quantitative easing policy, the Federal Reserve achieved incredible success in terms of ensuring increased consumer spending. However, the policy has a potential trouble. It almost tripled money supply in the economy. Such a move can increase inflation if situation of rapid economic growth occurs between 2014 and 2015. If the rapid economic growth does not occur in the near future, the US should focus on fiscal policies that seek to increase money that is available at the hands of the consumers. Besides, it can print more money. Rising consumer spending increases the aggregate demand whilst raising household incomes (OECD Economic Outlook, 2014).
Perhaps, quantitative easing strategies that were adopted by the US after the economic crisis were informed by this theoretical paradigm. If the government has been able to ensure a continuous economic growth at the rate of about 3.25%, maintenance of deficits worth half trillion dollars can be possible. Nevertheless, the government also needs to focus on reducing deficits by about $300 billion annually.
"""
times = []

print("Running inference 5 times...")

for i in range(5):
    start = time.time()
    summary = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False,
    truncation=True,
    max_new_tokens=60
)
    end = time.time()

    inference_time = end - start
    times.append(inference_time)

    print(f"Run {i+1}: {inference_time:.4f} seconds")

warm_times = times[1:]

print("\nWarm Average time:", sum(warm_times)/len(warm_times))


import torch
print("Model size (MB):", sum(p.numel() for p in summarizer.model.parameters()) * 4 / (1024**2))