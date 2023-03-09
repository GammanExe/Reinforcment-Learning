import prettytable as prettytable
from gym import Env
from gym.spaces import Box, MultiDiscrete
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy





class Data:
    ROOMS = [["Sala 1", 25], ["Sala 2", 45], ["Sala 3", 35], ["Sala 4", 30]]
    MEETING_TIMES = [["Termin 1", "Pon,Śr,Pt 09:00 - 10:00"],
                     ["Termin 2", "Pon,Śr,Pt 10:00 - 11:00"],
                     ["Termin 3", "Wt,Czw 09:00 - 10:30"],
                     ["Termin 4", "Wt,Czw 10:30 - 12:00"]]
    INSTRUCTORS = [["Prowadzący 1", "Dr James Web"],
                   ["Prowadzący 2", "Mr. Mike Brown"],
                   ["Prowadzący 3", "Dr Steve Day"],
                   ["Prowadzący 4", "Mrs Jane Doe"]]

    def __init__(self):
        self._rooms = [];
        self._meetingTimes = [];
        self._instructors = []
        for i in range(0, len(self.ROOMS)):
            self._rooms.append(Room(self.ROOMS[i][0], self.ROOMS[i][1]))
        for i in range(0, len(self.MEETING_TIMES)):
            self._meetingTimes.append(MeetingTime(self.MEETING_TIMES[i][0], self.MEETING_TIMES[i][1]))
        for i in range(0, len(self.INSTRUCTORS)):
            self._instructors.append(Instructor(self.INSTRUCTORS[i][0], self.INSTRUCTORS[i][1]))
        course1 = Course("Przedmiot 1", "Analiza", [self._instructors[0], self._instructors[1]], 25)
        course2 = Course("Przedmiot 2", "Progrmowanie", [self._instructors[0], self._instructors[1], self._instructors[2]], 35)
        course3 = Course("Przedmiot 3", "Algebra", [self._instructors[0], self._instructors[1]], 25)
        course4 = Course("Przedmiot 4", "Grafika komputerowa", [self._instructors[2], self._instructors[3]], 30)
        course5 = Course("Przedmiot5", "Bazy Danych", [self._instructors[3]], 35)
        course6 = Course("Przedmiot6", "Dynamika", [self._instructors[0], self._instructors[2]], 45)
        course7 = Course("Przedmiot7", "Fizyka Kwantowa", [self._instructors[1], self._instructors[3]], 45)
        self._courses = [course1, course2, course3, course4, course5, course6, course7]
        dept1 = Department("Dz. Matematyki", [course1, course3])
        dept2 = Department("Dz. Informatyki", [course2, course4, course5])
        dept3 = Department("Dz. Fizyki", [course6, course7])
        self._depts = [dept1, dept2, dept3]
        self._numberOfClasses = 0
        for i in range(0, len(self._depts)):
            self._numberOfClasses += len(self._depts[i].get_courses())

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_numberOfClasses(self):
        return self._numberOfClasses


class ScheduleEnv(Env):
    def __init__(self):


        self.action_space = MultiDiscrete([4, 4, 4])

        self.observation_space = Box(low =1,high =4  , shape=(7,3), dtype=int)
        self.index =0
        self.conflicts = 0

        self._data = data
        self._classes = []
        self._numbOfConflicts = 0
        self._fitness = -1
        self._classNumb = 0
        self._isFitnessChanged = True

        w, h = 2, 7
        self.courses_list = [[0 for x in range(w)] for y in range(h)]
        depts = self._data.get_depts()

        x = 0
        for i in range(0, len(depts)):

            courses = depts[i].get_courses()
            for j in range(0, len(courses)):
                self.courses_list[x][0] = depts[i]
                self.courses_list[x][1] = depts[i].get_courses()[j]
                x += 1

    def step(self, action):

        room=action[0]
        instructor=action[1]
        meeting_time=action[2]

        newClass = Class(self._classNumb, self.courses_list[self.index][0], self.courses_list[self.index][1])
        self._classNumb += 1

        self.schedule[self.index][0] = room+1
        self.schedule[self.index][1] = instructor+1
        self.schedule[self.index][2] = meeting_time+1

        newClass.set_room(data.get_rooms()[room])
        newClass.set_instructor(data.get_instructors()[instructor])
        newClass.set_meetingTime(data.get_meetingTimes()[meeting_time])
        self._classes.append(newClass)

        self.calculate_fitness()
        current_conflicts = self.get_numbOfConflicts()

        observation = self._get_obs()


        if self.conflicts < current_conflicts:

            reward = -1*(current_conflicts - self.conflicts)
            self.conflicts = current_conflicts

        else:
            reward = 1


        if self.index >= 6:
            done = True
        else:
            done = False

        info = {}

        self.index += 1

        return  observation, reward, done, info


    def step_showcase(self, action):

        room=action[0][0]
        instructor=action[0][1]
        meeting_time=action[0][2]

        newClass = Class(self._classNumb, self.courses_list[self.index][0], self.courses_list[self.index][1])
        self._classNumb += 1

        self.schedule[self.index][0] = room+1
        self.schedule[self.index][1] = instructor+1
        self.schedule[self.index][2] = meeting_time+1




        newClass.set_room(data.get_rooms()[room])
        newClass.set_instructor(data.get_instructors()[instructor])
        newClass.set_meetingTime(data.get_meetingTimes()[meeting_time])
        self._classes.append(newClass)

        self.calculate_fitness()
        current_conflicts = self.get_numbOfConflicts()



        observation = self.schedule



        if self.conflicts < current_conflicts:

            reward = -1*(current_conflicts - self.conflicts)
            self.conflicts = current_conflicts

        else:
            reward = 1


        if self.index >= 6:
            done = True
        else:
            done = False

        info = {}

        self.index += 1

        return  observation, reward, done, info ,self,self.conflicts



    def render(self):

        pass

    def reset(self):

        if len(self._classes) > 0:
            self._classes.clear()

        self.index = 0
        self.conflicts = 0
        self.schedule = np.zeros((7,3),dtype=int)



        observation = self.schedule

        return observation

    def _get_obs(self):
        return self.schedule

    def calculate_fitness(self):

        self._numbOfConflicts = 0

        classes = self.get_classes()

        for i in range(0, len(classes)):
            if classes[i].get_room().get_seatingCapacity() < classes[i].get_course().get_maxNumbOfStudents():
                self._numbOfConflicts += 1

            temp = []
            for x in range(0, len(classes[i].get_course().get_instructors())):
                temp.append(classes[i].get_course().get_instructors()[x])

            if classes[i].get_instructor() not in temp:
                self._numbOfConflicts += 1

            for j in range(0, len(classes)):
                if (j >= i):
                    if (classes[i].get_meetingTime() == classes[j].get_meetingTime() and
                            classes[i].get_id() != classes[j].get_id()):
                        if (classes[i].get_room() == classes[j].get_room()): self._numbOfConflicts += 1
                        if (classes[i].get_instructor() == classes[j].get_instructor()): self._numbOfConflicts += 1

    def get_classes(self):

        self._isFitnessChanged = True
        return self._classes

    def get_numbOfConflicts(self):
        return self._numbOfConflicts

    def get_fitness(self):

        if (self._isFitnessChanged == True):
            self._fitness = self.calculate_fitness()
            self._isFitnessChanged = False
        return self._fitness

    def __str__(self):
        returnValue = ""
        for i in range(0, len(self._classes) - 1):
            returnValue += str(self._classes[i]) + ", "
        returnValue += str(self._classes[len(self._classes) - 1])
        return returnValue

class Course:
    def __init__(self, number, name, instructors, maxNumbOfStudents):
        self._number = number
        self._name = name
        self._maxNumbOfStudents = maxNumbOfStudents
        self._instructors = instructors

    def get_number(self): return self._number

    def get_name(self): return self._name

    def get_instructors(self): return self._instructors

    def get_maxNumbOfStudents(self): return self._maxNumbOfStudents

    def __str__(self): return self._name


class Instructor:
    def __init__(self, id, name):
        self._id = id
        self._name = name

    def get_id(self): return self._id

    def get_name(self): return self._name

    def __str__(self): return self._name


class Room:
    def __init__(self, number, seatingCapacity):
        self._number = number
        self._seatingCapacity = seatingCapacity

    def get_number(self): return self._number

    def get_seatingCapacity(self): return self._seatingCapacity


class MeetingTime:
    def __init__(self, id, time):
        self._id = id
        self._time = time

    def get_id(self): return self._id

    def get_time(self): return self._time


class Department:
    def __init__(self, name, courses):
        self._name = name
        self._courses = courses

    def get_name(self): return self._name

    def get_courses(self): return self._courses


class Class:
    def __init__(self, id, dept, course):
        self._id = id
        self._dept = dept
        self._course = course
        self._instructor = None
        self._meetingTime = None
        self._room = None

    def get_id(self): return self._id

    def get_dept(self): return self._dept

    def get_course(self): return self._course

    def get_instructor(self): return self._instructor

    def get_meetingTime(self): return self._meetingTime

    def get_room(self): return self._room

    def set_instructor(self, instructor): self._instructor = instructor

    def set_meetingTime(self, meetingTime): self._meetingTime = meetingTime

    def set_room(self, room): self._room = room

    def __str__(self):
        return str(self._dept.get_name()) + "," + str(self._course.get_number()) + "," + \
               str(self._room.get_number()) + "," + str(self._instructor.get_id()) + "," + str(
            self._meetingTime.get_id())


class DisplayMgr:

    def print_schedule(self, schedule):
        table1 = prettytable.PrettyTable(
            ['Plan #', 'Ilość konfliktów', 'Przedmioty [Dział,Przedmiot,Sala,Prowadzący,Termin]'])

        table1.add_row([1, schedule.get_numbOfConflicts(),
                        schedule.__str__()])
        print(table1)

    def print_schedule_as_table(self, schedule):
        classes = schedule.get_classes()
        table = prettytable.PrettyTable(
            ['Indeks Przedmiotu ', 'Dział', 'Przedmiot (Indeks, max ilośc uczniów)', 'Sala (Pojemność)', 'Prowadzący (Indeks)',
             'Termin (Indeks)'])
        for i in range(0, len(classes)):
            table.add_row([str(i), classes[i].get_dept().get_name(), classes[i].get_course().get_name() + " (" +
                           classes[i].get_course().get_number() + ", " +
                           str(classes[i].get_course().get_maxNumbOfStudents()) + ")",
                           classes[i].get_room().get_number() + " (" + str(
                               classes[i].get_room().get_seatingCapacity()) + ")",
                           classes[i].get_instructor().get_name() + " (" + str(
                               classes[i].get_instructor().get_id()) + ")",
                           classes[i].get_meetingTime().get_time() + " (" + str(
                               classes[i].get_meetingTime().get_id()) + ")"])
        print(table)


data = Data()
displayMgr = DisplayMgr()
env = ScheduleEnv()

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

episodes = 5
for episode in range(1, episodes+1):
    observation = env.reset()

    done = False
    score = 0

    while not done:

        env.render()
        action = model.predict(observation)
        observation, reward, done, info, res_schedule, conflicts = env.step_showcase(action)
        score += reward
    print('Runda:{} Wynik kumulatywny:{} Ilość :{}'.format(episode, score,conflicts))

    displayMgr.print_schedule(res_schedule)
    displayMgr.print_schedule_as_table(res_schedule)

    print("\n")

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=350000)
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)

print(evaluate_policy(model, env, n_eval_episodes=1000, render=False))

# PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
#
# model = PPO.load(PPO_path, env=env)



for episode in range(1, episodes+1):
    observation = env.reset()

    done = False
    score = 0

    while not done:

        env.render()
        action = model.predict(observation)
        observation, reward, done, info, res_schedule, conflicts = env.step_showcase(action)
        score += reward
    print('Runda:{} Wynik kumulatywny:{} Ilość :{}'.format(episode, score,conflicts))

    displayMgr.print_schedule(res_schedule)
    displayMgr.print_schedule_as_table(res_schedule)


env.close()








