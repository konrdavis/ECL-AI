# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama

# This file makes use of a maximum of 5 prompts per function 
# to keep the model from going beyond the allowed memory.

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
    # Below are the sample prompts and responses provided to the model.
    tables = """
                    Prompt:
                    Given a dataset 'Orders' with fields 'orderID', 'customerName', and 'amount', create a record named 'orderRecord' that holds the customer name and the order ID. Then, use this record to make a table called 'OrderTable' holding the dataset.

                    Response:
                    orderRecord := RECORD
                        string      customerName := Orders.customerName;
                        int         orderID := Orders.orderID;
                    END;
                    OrderTable := TABLE(Orders,orderRecord);

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
                """
    transform = """
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
                """
    distribute = """
                    Prompt: Distribute the dataset repl_ind using wi as the key.

                    Response: dist_ind := DISTRIBUTE(repl_ind, wi);

                    Prompt: Distribute repl_wgt based on wi.

                    Response: dist_wgt := DISTRIBUTE(repl_wgt, wi);

                    Prompt: Distribute the concatenation of repl_ind and ind_ends across addr.

                    Response: dist_ind := DISTRIBUTE(repl_ind+ind_ends, addr);

                    Prompt: Distribute B_work across frst_addr.

                    Response: B_wrk0 := DISTRIBUTE(B_work, frst_addr);

                    Prompt: Distribute IndepDS using HASH32 with keys wi and id.

                    Response: dIndep  := DISTRIBUTE(IndepDS, HASH32(wi, id));
                """
    iterate = """
                    Prompt: Iterate over dataset xDatS, applying the setNewIds function to LEFT and RIGHT.

                    Response: xDat2 := ITERATE(xDatS, setNewIds(LEFT, RIGHT));

                    Prompt: Iterate over dataset splitPoints0, applying the doOneIter function to LEFT and RIGHT.

                    Response: splitPoints1 := ITERATE(splitPoints0, doOneIter(LEFT, RIGHT), LOCAL);

                    Prompt: Iterate over dataset with_st locally, applying the pass1 function to LEFT and RIGHT.

                    Response: prop_lnum := ITERATE(with_st, pass1(LEFT,RIGHT), LOCAL);

                    Prompt: Iterate over dataset w_x0, applying the checkAscending function to LEFT and RIGHT.

                    Response: w_x1 := ITERATE(w_x0, checkAscending(LEFT,RIGHT));

                    Prompt: Iterate over dataset raw_features, applying the vidMark function to LEFT and RIGHT. Keep only the resulting marked features.

                    Response: marked_features := ITERATE(raw_features, vidMark(LEFT,RIGHT))(keep_me);
                """
    dedup = """
                    Prompt: Dedup the dataset ohSorted using the first index and the LOCAL keyword.

                    Response: ohDedup := DEDUP(ohSorted, indexes[1], LOCAL);

                    Prompt: Dedup the dataset ds using the id3 field. Match against all records and assume there will be many duplicates.

                    Response: t := DEDUP(ds, id3, ALL, MANY);

                    Prompt: Dedup the dataset Val2Sort when LEFT and RIGHT Value2 fields are equal. Keep the last occurrence of each duplicate.

                    Response: Dedup4 := DEDUP(Val2Sort,LEFT.Value2 = RIGHT.Value2,RIGHT);

                    Prompt: Dedup the dataset Lasts when LEFT and RIGHT per_last_name fields are equal.

                    Response: MySet := DEDUP(Lasts,LEFT.per_last_name=RIGHT.per_last_name);

                    Prompt: Dedup the dataset raw using the id field locally. Keep the last occurrence of each duplicate.

                    Response: lastRaw := DEDUP(raw, id, RIGHT, LOCAL);
                """
    rollup = """
                    Prompt: ROLLUP the dataset grp_od, grouped, using the roll_d function on LEFT and ROWS(LEFT)).

                    Response: rslt := ROLLUP(grp_od, GROUP, roll_d(LEFT, ROWS(LEFT)));

                    Prompt: ROLLUP the dataset base_mod, grouped, using the pick1 function on ROWS(LEFT) for the rollup operation.

                    Response: base_rpt := ROLLUP(base_mod, GROUP, pick1(ROWS(LEFT)));

                    Prompt: ROLLUP the dataset grp_stats, grouped, using the roll_stats function with ROWS(LEFT) for the rollup operation.

                    Response: stats_block := ROLLUP(grp_stats, GROUP, roll_stats(ROWS(LEFT)));

                    Prompt: ROLLUP the dataset grp_named_coef, grouped, using the emod function on LEFT and ROWS(LEFT).

                    Response: ext_mod := ROLLUP(grp_named_coef, GROUP, emod(LEFT, ROWS(LEFT)));

                    Prompt: Rollup the dataset rr_ind, grouped, using the roll_part function on LEFT, ROWS(LEFT). Output record order is not significant.

                    Response: ind_mat := ROLLUP(rr_ind, GROUP, roll_part(LEFT, ROWS(LEFT), FALSE));
                """
    join = """
                    Prompt: Perform a JOIN operation between datasets st1 and st2 on LEFT.PersonID = RIGHT.PersonID. Apply the countem function to LEFT and RIGHT.

                    Response: j := JOIN(st1,st2,LEFT.PersonID=RIGHT.PersonID,countem(LEFT,RIGHT),LOCAL); 

                    Prompt: Perform a JOIN operation between datasets idx1 and idx2 on LEFT.FloatStr = RIGHT.DecStr. 

                    Response: base_rpt := ROLLUP(base_mod, GROUP, pick1(ROWS(LEFT)));

                    Prompt: ROLLUP the dataset grp_stats, grouped, using the roll_stats function with ROWS(LEFT) for the rollup operation.

                    Response: stats_block := ROLLUP(grp_stats, GROUP, roll_stats(ROWS(LEFT)));

                    Prompt: ROLLUP the dataset grp_named_coef, grouped, using the emod function on LEFT and ROWS(LEFT).

                    Response: ext_mod := ROLLUP(grp_named_coef, GROUP, emod(LEFT, ROWS(LEFT)));

                    Prompt: Rollup the dataset rr_ind, grouped, using the roll_part function on LEFT, ROWS(LEFT). Output record order is not significant.

                    Response: ind_mat := ROLLUP(rr_ind, GROUP, roll_part(LEFT, ROWS(LEFT), FALSE));
                """
    # Create a dialog to use the generation function of Llama3, adding prompts at the start.
    dialogs: List[Dialog] = []
    dialogs.append([{"role": "system", "content": "As an ECL code expert, you generate ECL code given a prompt in an effort to help the developer. Use these example prompts and responses to form your own answers. Only respond in code."},
                        {"role": "user", "content": tables + "\n" + sort + "\n" + transform + "\n" + distribute + "\n" + iterate + "\n" + dedup + "\n" + rollup + "\n" + join}])
    while True:
        # Ask for user input as long as the chat is running
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        # Add input to the created dialog and generate a response based on the current dialog
        dialogs[0].append({"role":"user", "content": "Remember to only Respond in ECL code \n Prompt: " + user_input})

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        # Provide the response to the user
        response = results[0]['generation']['content']
        print(f'ECLSeer: {response}\n')
        # Add the response to the dialog so the chatbot knows its previous responses.
        dialogs[0].append({"role": "assistant", "content":response})

if __name__ == "__main__":
    fire.Fire(main)
