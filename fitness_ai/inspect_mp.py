import mediapipe as mp
print('module file', mp.__file__)
print('dir', dir(mp))
print('solutions in dir?', 'solutions' in dir(mp))
print('tasks in dir?', 'tasks' in dir(mp))
print('tasks attributes:', dir(mp.tasks)[:50])
print('Pose attr in tasks?', hasattr(mp.tasks, 'Pose'))
