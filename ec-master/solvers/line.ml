open Core

open Client
open Timeout
open Task
open Utils
open Program
open Type

(* hurk need this random variable there to avoid "the first declaration uses unboxed representation" *)
type line_state = {pos : int; hitmp: int;}

(* tline_cont = state -> (state, list of blocks) *)
type line_cont = line_state -> line_state
let tline_cont = make_ground "line_cont";;

primitive "line_left" (tint @> tline_cont @> tline_cont)
         (let f : int -> line_cont -> line_cont = fun (d : int) ->
             fun (k : line_cont) ->
             fun (s : line_state) ->
               (*k {pos = s.pos - d}*)
               let s' = {s with pos = s.pos - d} in
               k s'
          in f);;
primitive "line_right" (tint @> tline_cont @> tline_cont)
         (let f : int -> line_cont -> line_cont = fun (d : int) ->
             fun (k : line_cont) ->
             fun (s : line_state) ->
               (*k {pos = s.pos + d}*)
               let s' = {s with pos = s.pos + d} in
               k s'
          in f);;

let evaluate_discrete_tower_program_LINE timeout p start =
    begin
      (* Printf.eprintf "%s\n" (string_of_program p); *)
      let p = analyze_lazy_evaluation p in
      let new_discrete =
        try
          match run_for_interval
                  timeout
                  (fun () -> run_lazy_analyzed_with_arguments p [fun s -> s] {pos=start; hitmp=1337})
          with
          | Some(p) ->
            Some(p.pos)
          | _ -> None
        with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
             (* we have to be a bit careful with exceptions *)
             (* if the synthesized program generated an exception, then we just terminate w/ false *)
             (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
             | otherException -> begin
                 if otherException = EnumerationTimeout then raise EnumerationTimeout else None
               end
      in
      new_discrete
    end
;;

register_special_task "SupervisedLine" (fun extra ?timeout:(timeout = 0.001)
    name task_type examples ->
  assert (task_type = tline_cont @> tline_cont);
  assert (examples = []);

  let open Yojson.Basic.Util in
  let start = extra |> member "start" |> to_int
  in
  let goal = extra |> member "goal" |> to_int
  in

  (* Printf.eprintf "TARGETING:\n%s\n\n" *)

  { name = name    ;
    task_type = task_type ;
    log_likelihood =
      (fun p ->
         let hit = (evaluate_discrete_tower_program_LINE timeout p start = Some(goal)) in
         (*Printf.eprintf "\t%s %b\n\n" (string_of_program p) hit;*)
         if hit then 0. else log 0.)
  })
;;
