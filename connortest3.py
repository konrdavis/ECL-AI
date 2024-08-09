# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct/",
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 3000,
    max_batch_size: int = 10,
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
    tables = """
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
             """
    transform = """
                    Prompt: Create a record using a transform function that takes a record L and an integer C as inputs, and returns a new record. The output record should have a field named Seq which is set to the value of C, and all other fields should be copied from the input record L. 

                    Response: OutRec XF2(Base L, INTEGER C) := TRANSFORM
                     SELF.Seq := C;
                     SELF := L;
                    END;

                    Prompt: Create a record using a transform function that takes a record L as input. The output fields should be 'float', calculated as L.PersonID / 1000 and 'dec', calculated as L.PersonID / 1000. All other fields should be taken from the input record.
                    
                    Response: r XF(r L) := TRANSFORM
                     SELF.float := L.PersonID / 1000;
                     SELF.dec := L.PersonID / 1000;
                     SELF := L;
                     
                    Prompt: Create a record using a transform function that takes a record L as input. The output fields should be 'surname', taken from L.sur, 'forename', taken from L.fore, and 'addr', taken from L.addr.

                    Response: namesRecord MoveData(ds L) := TRANSFORM
                    SELF.surname := L.sur;
                    SELF.forename := L.fore;
                    SELF.addr := L.addr;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. The output fields should be 'ranking', calculated as L.ranking + 1, with the rest of the fields being taken from R.

                    Response: rec RankGrpAccts(rec L, rec R) := TRANSFORM
                    SELF.Ranking := L.Ranking + 1;
                    SELF := R;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. The output fields should be 'ranking', calculated as L.ranking + 1 if L.PersonID is equal to R.PersonID, and 1 otherwise. The rest of the fields are taken from R.

                    Response: rec RankSrtAccts(rec L, rec R) := TRANSFORM
                    SELF.Ranking := IF(L.PersonID = R.PersonID,L.Ranking + 1, 1);
                    SELF := R;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. Take all the fields from L and R.

                    Response: r1 countem(t1 L, t2 R) := TRANSFORM
                    SELF := R;
                    SELF := L;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input.The output fields should be 'LeftLetter', taken from L.Letter, and 'RightLetter', taken from R.Letter.

                    Response: outrec joinEm(rec L, rec R) := TRANSFORM
                    SELF.LeftLetter := L.Letter;
                    SELF.RightLetter := R.Letter;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. Use the 'Tensor.R4.addSlices' method to add slices of L and R, storing the result in sumT. The output fields should be 'denseData', taken from sumT.denseData, and 'sparseData', taken from sumT.sparseData. The rest of the fields should be taken from L.

                    Response: t_Tensor doRollup(t_Tensor L, t_Tensor R) := TRANSFORM
                    sumT := Tensor.R4.addSlices(L, R);
                    SELF.denseData := sumT.denseData;
                    SELF.sparseData := sumT.sparseData;
                    SELF := L;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. The output field should be 'Sum', defined by L.value + R.value.

                    Response: additionCheck JoinThem(TensData L, TensData R) := TRANSFORM
                        SELF.Sum := L.value + R.value;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. The output should contain the field 'Value1', defined as L.Value1 if it is not an empty string; otherwise, set it to R.Value1. It should also contain the field 'LeftValue2', defined as L.Value2, and 'RightValue2', defined as R.Value2.

                    Response: MyOutRec JoinThem(MyRec L, MyRec R) := TRANSFORM
                        SELF.Value1 := IF(L.Value1<>'', L.Value1, R.Value1);
                        SELF.LeftValue2 := L.Value2;
                        SELF.RightValue2 := R.Value2;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and INTEGER C as input. The output should contain the field 'CatValues', defined by the concatination of L.Value1, L.Value2, '-', and C converted to a string. The rest of the fields are taken from L.

                    Response: MyOutRec CatThem(SomeFile L, INTEGER C) := TRANSFORM
                        SELF.CatValues := L.Value1 + L.Value2 + '-' + (STRING)C;
                        SELF := L;
                    END;

                    Prompt: Create a record using a transform function that takes a record t as input. The output should contain the field 'indexes', defined as the variable atIndex + t.indexes. The rest of the fields should be taken from t.

                    Response: Layout_Model2 extend_indexes(Layout_Model2 t) := TRANSFORM
                    SELF.indexes := atIndex + t.indexes;
                    SELF         := t;
                    END;

                    Prompt: Create a record using a transform function that takes a record le and UNSIGNED c as input. The output should contain the field 'number' defined as c. Create a variable 'order', defined as the integer division of c by 2. The field 'value' should be defined by LOG of le.value multiplied by POWER(le.value, order) if c is odd, and POWER(le.value,order) otherwise. The rest of the fields should be taken from le.

                    Response: Types.NumericField mn(Types.NumericField le,UNSIGNED c) := TRANSFORM
                    SELF.number := c;
                    order := c DIV 2;
                    SELF.value := IF ( c & 1 = 1, LOG(le.value), 1 ) * POWER(le.value,order);
                    SELF := le;
                    END;

                    Prompt: Create a record using a transform function that takes a record le and a record ri as input. The output should contain the field 'Pos', defined by ri.pos, with the rest of the fields being taken from le.

                    Response: AveRanked Into(D le,T ri) := TRANSFORM
                        SELF.Pos := ri.pos;
                        SELF := le;
                    END;

                    Prompt: Create a record using a transform function that takes a record L and a record R as input. The output should contain the field 'Concordant', defined by 1 if (L.value < R.value AND L.value2 < R.value2) OR (L.value > R.value AND L.value2 > R.value2), and 0 otherwise. It should also contain the field 'Discordant', defined by 1 if (L.value < R.value AND L.value2 > R.value2) OR (L.value > R.value AND L.value2 < R.value2), and 0 otherwise.

                    Response: KendallCompRec KendallComp(combineRec L, combineRec R) := TRANSFORM
                        SELF.Concordant := IF((L.value < R.value AND L.value2 < R.value2) OR
                                            (L.value > R.value AND L.value2 > R.value2),
                                            1, 0);
                        SELF.Discordant := IF((L.value < R.value AND L.value2 > R.value2) OR
                                            (L.value > R.value AND L.value2 < R.value2),
                                            1, 0);
                    END;

                    Prompt: Create a record using a transform function that takes a record le and a UNSIGNED C as input. The output should contain the fields 'wi' defined as le.wi, 'value' defined as the variable v, 'id' defined as le.id, and 'number' defined as C.

                    Response: Types.NumericField bv(seeds le,UNSIGNED C) := TRANSFORM
                    SELF.wi := le.wi;
                    SELF.value := v;
                    SELF.id := le.id;
                    SELF.number := C;
                    END;

                    Prompt: Create a record using a transform function that takes a record le and a UNSIGNED c as input. The output should contain the fields '__node' defined as ThorLib.node(), 'seq' defined as c, and the rest of the fields should be taken from le.

                    Response: LOCAL extend_rec add_rank(infile le, UNSIGNED c) := TRANSFORM
                        SELF.__node := ThorLib.node();
                        SELF.seq := c;
                        SELF := le;
                    END;

                    Prompt: Create a record using a transform function that takes a UNSIGNED x as input. The output should contain the fields 'wi' defined by (x-1) DIV (num_samples) + 1, 'id' defined as x, and 'label' defined by RANDOM() % num_labels.

                    Response: Labels GenerateSample(UNSIGNED x) := TRANSFORM
                    SELF.wi := (x-1) DIV (num_samples) + 1;
                    SELF.id := x;
                    SELF.label := RANDOM() % num_labels;
                    END;

                    Prompt: Create a record using a transform function that takes a INTEGER x as input. The output should contain the fields 'wi' defined as (X-1) DIV (num_samples * num_variables) + 1, 'id' defined as (x - 1) DIV (num_variables) + 1, 'number' defined as (x - 1) % num_variables + 1, and 'value' defined as (RANDOM()) % (num_classes).

                    Response: 

                    Prompt: Create a record using a transform function that takes a UNSIGNED x as input. The output should contain the fields 'wi' defined as (X-1) DIV (num_samples * num_dimensions) + 1, 'id' defined as (x - 1) DIV (num_dimensions) + 1, 'number' defined by (x - 1) % num_dimensions + 1, and 'value' defined by (RANDOM() % 100)/100.

                    Response: NumericField GenerateSample(UNSIGNED x) := TRANSFORM
                    SELF.wi := (x-1) DIV (num_samples * num_dimensions) + 1;
                    SELF.id := (x-1) DIV (num_dimensions) + 1;
                    SELF.number := (x-1) % num_dimensions + 1;
                    SELF.value := (RANDOM() % 100)/100;
                    END;

                    Prompt: Create a record using a transform function that takes a record le and dataset ri as input. The output should contain the field 'value03' defined as SQRT(SUM(ri,value03)) and the fields 'number', 'value01', and 'value02', all equal to 0. The rest of the fields should be taken from le.

                    Response: ClusterPair take(ClusterPair le, DATASET(ClusterPair) ri) := TRANSFORM
                            SELF.value03 := SQRT(SUM(ri,value03));
                            SELF.number := 0;
                            SELF.value01 := 0;
                            SELF.value02 := 0;
                            SELF := le;
                        END;

                    Prompt: Create a record using a transform function that takes a record obs and record vars as input. The output should contain the fields 'wi' defined as obs.wi, 'df' defined as obs.value - vars.value - 1, and 'ind_vars' defined as vars.value.

                    Response: ivdf_rec make_ivdf(Model obs, Model vars) := TRANSFORM
                    SELF.wi := obs.wi;
                    SELF.df := obs.value - vars.value - 1;
                    SELF.ind_vars := vars.value;
                    END;

                    Prompt: Create a record using a transform function that takes a record rq and UNSIGNED c as input. The output should contain the field 'rq_nominal' defined as (c - 1) * ThorLib.nodes() + 1 + ThorLib.node(). The rest of the fields should be taken from rq.

                    Response: ex_rq enum_rq(Types.LUCI_Model_Rqst rq, UNSIGNED c) := TRANSFORM
                        SELF.rq_nominal := (c-1)*ThorLib.nodes() + 1 + ThorLib.node();
                        SELF := rq;
                    END;

                    Prompt: Create a record using a transform function that takes a record x and record b as input. The output should contain the fields 'wi' defined as x.wi, 'id' defined as x.id, 'number' defined as b.dep_nom, and 'raw' defined as x.value * b.w

                    Response: Raw_Prediction mult(NumericField x, Model_Coef b) := TRANSFORM
                        SELF.wi := x.wi;
                        SELF.id := x.id;
                        SELF.number := b.dep_nom;
                        SELF.raw := x.value*b.w;
                    END;
             """
    sort = """
                    Prompt: Sort the dataset PeopleStatsDataset by record.

                    Response: SortedTable := SORT(PeopleStatsDataset, RECORD);

                    Prompt: Sort the dataset ML.Regression.Sparse.OLS_Cholesky(X,Y).Extrapolated(X) by the id field.

                    Response: sparse_extrapo_height:=sort(ML.Regression.Sparse.OLS_Cholesky(X,Y).Extrapolated(X),id);

                    Prompt: Sort the dataset ML.Regression.Dense.OLS_Cholesky(X,Y).Extrapolated(X) by the id field.

                    Response: dense_extrapo_height:=sort(ML.Regression.Dense.OLS_Cholesky(X,Y).Extrapolated(X),id);

                    Prompt: Concatenate the datasets armg, roag, romc, rbmc, rpmg, and recg and sort the result by the arm_group_id field.

                    Response: all_recs := SORT(armg+roag+romc+rbmc+rpmg+recg, arm_group_id);

                    Prompt: Sort the dataset tab projected onto Base_Tab by the hasID field in descending order.

                    Response: SELF.s_tab := SORT(PROJECT(tab, Base_Tab), hasID, -c);

                    Prompt: Sort the dataset tbl by PersonID in ascending order, Opendate in descending order, and Balance in ascending order.

                    Response: SortRecs := SORT(tbl,PersonID,-Opendate,-Balance);

                    Prompt: Sort the dataset ds0 by personid in ascending order, opendate in ascending order, and balance in descending order.

                    Response: s1 := SORT(ds0,personid,opendate,-balance);

                    Prompt: Sort the dataset ds1 by personid in ascending order, opendate in ascending order, and balance in descending order locally.

                    Response: s3 := SORT(ds1,personid,opendate,-balance,LOCAL);

                    Prompt: Sort the dataset ds1 by personid in ascending order locally.

                    Response: s4 := SORT(ds1,personid,LOCAL);

                    Prompt: Sort the dataset x excluding the field sex.

                    Response: y := SORT(x,EXCEPT sex);

                    Prompt: Sort the dataset LastTbl by the field per_last_name in ascending order.

                    Response: Lasts := SORT(LastTbl,per_last_name);

                    Prompt: Sort the dataset NamesTbl1 by per_last_name in ascending order first, and then by per_first_name in ascending order.

                    Response: Names1 := SORT(NamesTbl1,per_last_name,per_first_name);

                    Prompt: Sort the dataset NamesTbl by per_last_name in ascending order first, and then by per_first_name in ascending order.

                    Response: Names2 := SORT(NamesTbl,per_last_name,per_first_name);

                    Prompt: Sort the dataset GroupedSet by the field first_name in ascending order.

                    Reponse: SecondSort := SORT(GroupedSet,first_name);

                    Prompt: Sort the dataset my_dataset by the field strkey in ascending order.

                    Response: my_sorted := SORT(my_dataset, strkey);

                    Prompt: Sort the dataset scored by the fields wi, classifier, actual_class, and predict_class in ascending order.

                    Response: srt_dtl := SORT(scored, wi, classifier, actual_class, predict_class);

                    Prompt: Sort the dataset l1_dep by the fields number and value in ascending order.

                    Response: l1_dep_srt := SORT(l1_dep, number , value);

                    Prompt: Sort the dataset w_rq by the fields rq_nominal and model_id in ascending order.

                    Response: srted_w_rq := SORT(w_rq, rq_nominal, model_id);

                    Prompt: Sort the dataset detail by the fields wi and classifier in ascending order.

                    Response: srt_details := SORT(detail, wi, classifier);

                    Prompt: Sort the concatenated datasets dist_ind and end_ind by the fields wi, number, id, and dropMe locally.

                    Response: srtd_ind := SORT(dist_ind+end_ind, wi, number, id, dropMe, LOCAL);

                    Prompt: Sort the dataset dist_ind by the fields wi, part, number, id, and dropMe locally.

                    Response: sorted_ind := SORT(dist_ind, wi, part, number, id, dropMe, LOCAL);

                    Prompt: Sort the dataset B_dist by the fields wi and this_addr locally.

                    Response: B_init := SORT(B_dist, wi, this_addr, LOCAL);

                    Prompt: Sort the result of applying the rslt function to the err dataset by the fields test_name, c, and r.

                    Response: errors := SORT(rslt(err), test_name, c, r);

                    Prompt: Sort the result of the trades dataset filtered by ValidDate(trades.trd_dopn) and isMortgage, then sort the filtered dataset by the field trades.trd_dopn_mos.

                    Response: SortedTrades := SORT(trades(ValidDate(trades.trd_dopn),isMortgage),
                                trades.trd_dopn_mos);
                    
                    Prompt: Sort the dataset f0 by the field ip_from in ascending order, and assign the sorted result to fs.

                    Response: fs := SORT(f0,ip_from);
        """

    dialogs: List[Dialog] = []
    i = 0
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        dialogs.append([{"role": "system", "content": "As an ECL code expert, you generate ECL code given a prompt in an effort to help the developer. Use these example prompts and responses to form your own answers. Only respond in code."},
                        {"role": "user", "content": tables + "\n" + sort + "\n Prompt: " + user_input}])

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        response = results[i]['generation']['content']
        print(f'Assistant: {response}\n')

        dialogs[i].append({"role": "assistant", "content":response})
        i += 1


if __name__ == "__main__":
    fire.Fire(main)
