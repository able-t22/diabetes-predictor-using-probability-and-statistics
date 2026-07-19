def generate_report(prob1,prob0):
    print()
    print("Probability of Diabetes")
    print(round(prob1*100,2),"%")
    print()
    print("Probability of Non Diabetes")
    print(round(prob0*100,2),"%")
    print()
    if prob1>prob0:
        print("Prediction : Diabetic")
    else:
        print("Prediction : Non Diabetic")
