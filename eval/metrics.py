import numpy as np 
import pandas as pd 
import sys

class IncrementalTracker:
    def __init__(self):
        self.most_recent = {}
        self.greatest_past = {}
        self.start_indexes = {} # Recall@1 for latest 
        self.seen_envs = []

    # dictionary 형태로 각 env에 대한 성능 결과값을 받아 저장
    def update(self, update_dict, env_idx):
        for k,v in update_dict.items():
            
            # 이미 관측한 환경이 아닐 경우
            if k not in self.seen_envs:
                self.seen_envs.append(k)
                
                # 처음본 환경 == 가장 최근 환경
                self.most_recent[k] = v  
                
                self.greatest_past[k] = np.nan 
                
                # 해당 환경이 몇번째에서 부터 학습되는지 저장
                self.start_indexes[k] = env_idx 
                
            # 이미 관측한 환경일 경우
            else:
                # nan일 경우 most_recent에 저장된 v로 greateset past를 저장하고, 아니면 max값을 저장 
                self.greatest_past[k] = self.most_recent[k] if np.isnan(self.greatest_past[k]) else max(self.greatest_past[k], v)
                
                # most_recent 업데이트
                self.most_recent[k] = v 

    def get_results(self):
        # Get recall and forgetting
        results = {}
        
        # dictionary에서 for문으로 하나만 받으면 key만 나옴
        for k in self.start_indexes:
            results[k] = {}
            results[k]['Recall@1'] = self.most_recent[k]
            if k in self.greatest_past:
                results[k]['Forgetting'] = self.greatest_past[k] - self.most_recent[k]
            else:
                results[k]['Forgetting'] = np.nan
        
        # Merge
        results_merged = {} 
        for v in self.start_indexes.values():
            # v번째에 맞는 환경을 가져오기
            merge_keys = [k for k in self.start_indexes if self.start_indexes[k] == v] # Get keys which should be merged 
            print("merge_keys : ", merge_keys)
            # indexing 없이 원소만 빼내려고 이렇게 하는듯? + str변환
            new_key = '/'.join(merge_keys) # Get new key 
            merged_recall = np.mean([results[m]['Recall@1'] for m in merge_keys])
            merged_forgetting = np.mean([results[m]['Forgetting'] for m in merge_keys])

            results_merged[new_key] = {'Recall@1': merged_recall, 'Forgetting': merged_forgetting}
        
        # Print 
        results_final = pd.DataFrame(columns = ['Recall@1', 'Forgetting'])
        for k in results_merged:
            results_final.loc[k] = [results_merged[k]['Recall@1'], results_merged[k]['Forgetting']]
        results_final.loc['Average'] = results_final.mean(0)
        return results_final 

if __name__ == '__main__':
    step_0 = {'Oxford': 93.8}
    step_1 = {'Oxford': 84.09552543691579, 'DCC': 81.06894504403417}
    step_2 = {'Oxford': 83.91506599787158, 'DCC': 77.28090467849917, 'Riverside': 82.6737801326445}
    step_3 = {'Oxford': 82.93509829383399, 'DCC': 68.0391049140116, 'Riverside': 76.65990835042547, 'In-house': 95.75022601898989}

    metrics = IncrementalTracker()

    metrics.update(step_0, 0)
    metrics.update(step_1, 1)
    metrics.update(step_2, 2)
    metrics.update(step_3, 3)

    r = metrics.get_results()
    print(r)