---
icon: terminal
tags: [guide]
order: 40
---

# 4. Checking Training Results

앞 장과 같이 `train_llama2.py` 스크립트를 실행하면 결과 모델이 `llama2_summarization` 디렉토리에 저장됩니다. 이는 순수한 PyTorch 모델 파라미터 파일로 MoAI Platform이 아닌 일반 GPU 서버에서도 완벽하게 호환됩니다.

미리 다운로드한 GitHub 레포지토리의 `tutorial` 디렉토리 아래에 있는 `inference_llama2.py` 스크립트로 학습된 모델을 테스트해 볼 수 있습니다. 

테스트에는 영국 프리미어 리그(EPL) 경기 결과와 관련된 기사 내용이 사용되었습니다.

```python
# tutorial/inference_llama2.py
...
input_text = """[SUMMARIZE] (CNN)Arsenal kept their slim hopes of winning this season's English Premier League title alive by beating relegation threatened Burnley 1-0 at Turf Moor. A first half goal from Welsh international Aaron Ramsey was enough to separate the two sides and secure Arsenal's hold on second place. More importantly it took the north London club to within four points of first placed Chelsea, with the two clubs to play next week. But Chelsea have two games in hand and play lowly Queens Park Rangers on Sunday, a team who are themselves struggling against relegation. Good form . Arsenal have been in superb form since the start of the year, transforming what looked to be another mediocre season struggling to secure fourth place -- and with it Champions League qualification -- into one where they at least have a shot at winning the title. After going ahead, Arsenal rarely looked in any danger of conceding, showing more of the midfield pragmatism epitomized by the likes of Francis Coquelin, who also played a crucial role in the goal. "He has been absolutely consistent in the quality of his defensive work," Arsenal coach Arsene Wenger told Sky Sports after the game when asked about Coquelin's contribution to Arsenal's current run. They have won eight games in a row since introducing the previously overlooked young Frenchman into a more defensive midfield position. "He was a player who was with us for seven years, from 17, he's now just 24," Wenger explained. "Sometimes you have to be patient. I am very happy for him because he has shown great mental strength." Now all eyes will be on next week's clash between Arsenal and Chelsea which will likely decide the title. "They have the games in hand," said Wenger, playing down his club's title aspirations. "But we'll keep going and that's why the win was so important for us today." Relegation dogfight . Meanwhile it was a good day for teams at the bottom of the league. Aston Villa continued their good form since appointing coach Tim Sherwood with a 1-0 victory over Tottenham, who fired Sherwood last season. Belgian international Christian Benteke scored the only goal of the game, his eighth in six matches, to secure a vital three points to give the Midlands club breathing space. Another Midlands club looking over their shoulder is West Brom, who conceded an injury time goal to lose 3-2 against bottom club Leicester City. But it was an awful day for Sunderland's former Dutch international coach Dick Advocaat, who saw his team lose 4-1 at home against form team Crystal Palace. Democratic Republic of Congo international Yannick Bolasie scored Crystal Palace's first ever hat trick in the Premier League to secure an easy victory. [/SUMMAIRZE]"""
```

코드를 실행합니다.

```bash
~/quickstart$ python tutorial/inference_llama2.py
```

출력값을 확인해보면 Llama2가 프롬프트의 내용을 적절하게 요약한 것을 확인할 수 있습니다.

``` bash
Llama2: [SUMMARIZE] (CNN)Arsenal kept their slim hopes of winning this season's English Premier League title alive by beating relegation threatened Burnley 1-0 at Turf Moor. A first half goal from Welsh international Aaron Ramsey was enough to separate the two sides and secure Arsenal's hold on second place. More importantly it took the north London club to within four points of first placed Chelsea, with the two clubs to play next week. But Chelsea have two games in hand and play lowly Queens Park Rangers on Sunday, a team who are themselves struggling against relegation. Good form . Arsenal have been in superb form since the start of the year, transforming what looked to be another mediocre season struggling to secure fourth place -- and with it Champions League qualification -- into one where they at least have a shot at winning the title. After going ahead, Arsenal rarely looked in any danger of conceding, showing more of the midfield pragmatism epitomized by the likes of Francis Coquelin, who also played a crucial role in the goal. "He has been absolutely consistent in the quality of his defensive work," Arsenal coach Arsene Wenger told Sky Sports after the game when asked about Coquelin's contribution to Arsenal's current run. They have won eight games in a row since introducing the previously overlooked young Frenchman into a more defensive midfield position. "He was a player who was with us for seven years, from 17, he's now just 24," Wenger explained. "Sometimes you have to be patient. I am very happy for him because he has shown great mental strength." Now all eyes will be on next week's clash between Arsenal and Chelsea which will likely decide the title. "They have the games in hand," said Wenger, playing down his club's title aspirations. "But we'll keep going and that's why the win was so important for us today." Relegation dogfight . Meanwhile it was a good day for teams at the bottom of the league. Aston Villa continued their good form since appointing coach Tim Sherwood with a 1-0 victory over Tottenham, who fired Sherwood last season. Belgian international Christian Benteke scored the only goal of the game, his eighth in six matches, to secure a vital three points to give the Midlands club breathing space. Another Midlands club looking over their shoulder is West Brom, who conceded an injury time goal to lose 3-2 against bottom club Leicester City. But it was an awful day for Sunderland's former Dutch international coach Dick Advocaat, who saw his team lose 4-1 at home against form team Crystal Palace. Democratic Republic of Congo international Yannick Bolasie scored Crystal Palace's first ever hat trick in the Premier League to secure an easy victory. [/SUMMAIRZE]
Arsenal beat Burnley 1-0 in the English Premier League.
Aaron Ramsey scores the only goal of the game.
Arsenal remain in second place.
Chelsea can extend their lead to seven points.
```

