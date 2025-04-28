import pandas as pd

class GlobalTracker:
    def __init__(self):
        self.logs = {}
        self.agents = {}
        self.distributions = {}
        self.exit_sim = False

    def log_decision(self, logging_content, sim_nr):
        key = f"Sim_{sim_nr}"
        if key in self.logs:
            self.logs[key].append(logging_content)
        else:
            self.logs[key] = [logging_content]

    def save(self, file_path):
        # Flatten the nested dictionary for DataFrame conversion
        # Might need to get the relevant features from the personas here
            # make own columns, then melt to feature column
        def extract_features(persona):
            # Assuming persona has a 'features' attribute formatted as described
            feature_list = [(name, value[0], value[1]) for name, value in persona.scratch.features]
            return feature_list
        
        flattened_logs = []
        for sim_nr, logs in self.logs.items():
            for log in logs:
                hiring_persona = log["Init"]
                employee_persona = log["Target"]
                if hiring_persona:
                    hiring_features = extract_features(hiring_persona)
                    hiring_name = hiring_persona.scratch.name
                    if employee_persona:
                        empl_features = extract_features(employee_persona)
                        empl_name = employee_persona.scratch.name
                    else:
                        empl_features = [(h_feature_name, None, None) for h_feature_name, _, _ in hiring_features]
                        empl_name = ""
                    additional_info = {k: v for k, v in log.items() if k not in ["Init", "Target"]}
                    for (h_feature_name, h_feature_value, h_feature_range), (e_feature_name, e_feature_value, e_feature_range) in zip(hiring_features, empl_features):
                        flattened_log = {
                            "SIM_NR": sim_nr,
                            "init": hiring_name,
                            "target": empl_name,
                            "h_feature_name": h_feature_name,
                            "h_feature_value": h_feature_value,
                            "h_feature_range": h_feature_range,
                            "e_feature_name": e_feature_name,
                            "e_feature_value": e_feature_value,
                            "e_feature_range": e_feature_range
                            }
                        flattened_log.update(additional_info)
                        flattened_logs.append(flattened_log)
                    if not flattened_log:
                        flattened_log = {
                            "SIM_NR": sim_nr,
                            "init": hiring_name,
                            "target": empl_name,
                            }
                        flattened_log.update(additional_info)
                        flattened_logs.append(flattened_log)
            
        df = pd.DataFrame(flattened_logs)
        df.to_csv(file_path, index=False)
