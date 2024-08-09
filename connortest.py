# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
    prompt: str,
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct/",
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": "As an ECL code expert, you generate ECL code given a prompt in an effort to help the developer. Use these example prompts and responses to form your own answers. Only respond in code."},
            {"role": "user", "content": """
                    Prompt:
                    Given a dataset 'names' with fields 'surname', 'forename', and 'age', create a table named 'newTable' including these fields, grouped by surname. 

                    Response:
                    newTable := TABLE(names,{surname, forename, age},surname);

                    Prompt:
                    Given a dataset 'Names' and a record 'NamesRecord', create a table 'NamesTable' using this dataset and record.

                    Response:
                    NamesTable := TABLE(Names, NamesRecord);

                    Prompt:
                    Given a dataset 'Person' and a record 'R', use the fields 'per_state' and 'per_sex' to create a table 'PerState' grouped by state and sex.

                    Response:
                    PerState:= TABLE(Person,R, per_st, per_sex);

                    Prompt:
                    Given a dataset 'NamesData' with fields 'forename' and 'surname', create a record named 'namesRecord' that holds the full name (forename + surname), the forename, and the surname. Then, use this record to make a table called 'NamesTable' holding the dataset.

                    Response:
                    namesRecord :=
                    RECORD
                    string          fullname := namesData.forename + ' ' + namesData.surname;
                    string          forename := namesData.forename;
                    string          surname := namesData.surname;
                                END;
                    NamesTable := TABLE(namesData,namesRecord);

                    Prompt:
                    Given a dataset 'Places' with fields 'city', 'state', and 'address', create a record named 'placesRecord' that holds the city and the state. Then, use this record to make a table called 'placesTable' holding the dataset.

                    Response:
                    namesRecord :=
                    RECORD
                    string          city := Places.city;
                    string          state := Places.state;
                                END;
                    placesTable := TABLE(Places,placesRecord);
             
                    Prompt: 
                    Given a dataset 'Things' with fields 'height', 'length', and 'width', create record with the fields length and width. Using this record a table named 'thingsTable' to hold the dataset.

                    Response:
                    thingsRecord :=
                    RECORD
                    decimal length := Things.length;
                    decimal width := Things.width;
                    END;
                    thingsTable := TABLE(Things,thingsRecord);
             
                    Prompt:
                    Given a dataset 'Places' with fields 'city', 'state', and 'address', create a table called 'placesTable' holding the dataset with fields 'state' and 'city'.

                    Response:
                    placesTable := TABLE(Places, {state, city});

                    Prompt:
                    Given a dataset 'NamesData' with fields 'forename' and 'surname', create a record named 'namesRecord' that holds the forename and the surname. Then, use this record to make a table called 'NamesTable' holding the dataset, grouped by surname.

                    Response:
                    "namesRecord :=
                    RECORD
                    string          forename := namesData.forename;
                    string          surname := namesData.surname;
                                END;
                    NamesTable := TABLE(namesData,namesRecord,surname);"

                    Prompt:
                    Given a dataset 'Employees' with fields 'department', 'position', and 'salary', create a table named 'DeptTable' including these fields, grouped by department.

                    Response:
                    deptTable := TABLE(Employees, {department, position, salary}, department);

                    Prompt:
                    Given a dataset 'Students' with fields 'grade', 'class', and 'name', create a table named 'GradeTable' including these fields, grouped by grade.

                    Response:
                    GradeTable := TABLE(Students, {grade, class, name}, grade);

                    Prompt:
                    Given a dataset 'Employees' with fields 'department', 'position', and 'salary', create a record named 'employeeRecord' that holds the department and the position. Then, use this record to make a table called 'EmployeeTable' holding the dataset.

                    Response:
                    "employeeRecord :=
                    RECORD
                    string          department := Employees.department;
                    string          position := Employees.position;
                                END;
                    EmployeeTable := TABLE(Employees,employeeRecord);"

                    Prompt:
                    Given a dataset 'Books' with fields 'author', 'title', and 'publication_year', create a table named 'BooksTable' including these fields, grouped by author.

                    Response:
                    BooksTable := TABLE(Books, {author, title, publication_year}, author);

                    Prompt:
                    Given a dataset 'Vehicles' with fields 'make', 'model', and 'year', create a record named 'vehicleRecord' that holds the make and the model. Then, use this record to make a table called 'VehicleTable' holding the dataset.

                    Response:
                    "vehicleRecord :=
                    RECORD
                    string          make := Vehicles.make;
                    string          model := Vehicles.model;
                                END;
                    ProductTable := TABLE(Vehicles,vehicleRecord);"

                    Prompt:
                    Given a dataset 'Products' with fields 'category', 'name', and 'price', create a table named 'ProductsTable' including the fields name and price.

                    Response:
                    ProductsTable := TABLE(Products, {name, price});

                    Prompt:
                    Given a dataset 'Companies' with fields 'industry', 'name', and 'revenue', create a table named 'CompaniesTable' including these fields, grouped by industry.

                    Response:
                    CompaniesTable := TABLE(Companies, {industry, name, revenue}, industry);

                    Prompt:
                    Given a dataset 'Events' with fields 'type', 'location', and 'date', create a table named 'EventsTable' including the fields location and date.

                    Response:
                    EventsTable := TABLE(Events, {location, date});

                    Prompt:
                    Given a dataset 'Inventory' with fields 'warehouse', 'item', and 'quantity', create a table named 'WarehouseInventory' including these fields, grouped by warehouse.

                    Response:
                    WarehouseInventory := TABLE(Inventory, {warehouse, item, quantity}, warehouse);

                    Prompt:
                    Given a dataset 'Orders' with fields 'orderID', 'customerName', and 'amount', create a record named 'orderRecord' that holds the customer name and the order ID. Then, use this record to make a table called 'OrderTable' holding the dataset.

                    Response:
                    "orderRecord :=
                    RECORD
                    string          customerName := Orders.customerName;
                    int                orderID := Orders.orderID;
                                END;
                    OrderTable := TABLE(Orders,orderRecord);"

                    Prompt:
                    Given a dataset 'Movies' with fields 'director', 'title', and 'release_year', create a table named 'MoviesTable' including the fields title and release_year.

                    Response:
                    MoviesTable := TABLE(Movies, {title, release_year});

                    Prompt:
                    Given a dataset 'Employees' with fields 'employee_id', 'department', and 'salary', create a table named 'DeptSalaries' holding the dataset with fields 'department' and 'salary'.

                    Response:
                    DeptSalaries := TABLE(Employees, {department, salary});

                    Prompt:
                    Given a dataset 'Customers' with fields 'customer_id', 'name', and 'email', create a table named 'CustomerInfo' holding the dataset with fields 'name' and 'email'.

                    Response:
                    CustomerInfo := TABLE(Customers, {name, email});

                    Prompt:
                    Given a dataset 'Vehicles' with fields 'make', 'model', and 'year', create a table named 'VehicleDetails' holding the dataset with these fields, grouped by make.

                    Response:
                    VehicleDetails := TABLE(Vehicles, {make, model, year}, make);

                    Prompt:
                    Given a dataset 'Projects' with fields 'project_name', 'manager', and 'budget', create a table named 'ProjectsTable' holding the dataset with fields 'manager' and 'project_name'.

                    Response:
                    ProjectsTable := TABLE(Projects, {manager, project_name});

                    Prompt"
             """ + prompt + """
              Response: """},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
