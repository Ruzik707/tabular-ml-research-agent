import json

from automl_agent.agent import compact_recipe_history, recipe_feedback


def test_recipe_history_keeps_parameters_and_errors_without_duplicate_diagnostics():
    proposal={'base_trial_id':'trial_0001','hypothesis':'test','parameters':{'iterations':500},
              'fit_options':{'early_stopping_rounds':100},'decision':{'rationale':'long text '*2000}}
    result={'ok':True,'value':{'selected_trial':'trial_0001','best_metric':.6,'trials':[
        {'trial_id':'trial_0002','status':'ok','metrics':{'macro_f1':.55},'diagnostics':{'large':['facts']*2000}},
        {'status':'invalid_recipe','error':'auto_class_weights is classification-only'}]}}
    step=[{'role':'assistant','content':'','tool_calls':[{'id':'c1','type':'function','function':{
        'name':'run_model_experiments','arguments':json.dumps({'experiments':[proposal]})}}]},
        {'role':'tool','tool_call_id':'c1','name':'run_model_experiments','content':json.dumps(recipe_feedback(result))}]
    before=json.dumps(step)
    compact=compact_recipe_history(step,{'run_model_experiments':{'type':'object','properties':{
        'experiments':{'type':'array','items':{'type':'object'}}}}})
    assert json.dumps(step)==before
    args=json.loads(compact[0]['tool_calls'][0]['function']['arguments'])
    assert args['experiments'][0]['parameters']=={'iterations':500}
    assert args['experiments'][0]['fit_options']=={'early_stopping_rounds':100}
    assert 'decision' not in args['experiments'][0]
    assert len(json.dumps(compact))<1600
    assert 'classification-only' in compact[1]['content']
    assert result['value']['trials'][0]['diagnostics']['large']
