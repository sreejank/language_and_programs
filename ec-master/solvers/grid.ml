open Core
open Array

open Client
open Timeout
open Task
open Utils
open Program
open Type

exception Exception of string


type dir_type =
{
	delta_x : int;
	delta_y : int;
}
let up = {delta_x = -1; delta_y = 0};;
let down = {delta_x = 1; delta_y = 0};;
let left = {delta_x = 0; delta_y = -1};;
let right = {delta_x = 0; delta_y = 1};;

type grid_state =
{
	mutable x : int;
	mutable y : int;
	mutable dir : dir_type;
	mutable pendown : bool;
	mutable reward : float;
	w : int;
	h : int;
	mutable board : (bool array) array;
	cost_pen_change : bool;
	cost_when_penup : bool;
};;

type grid_cont = grid_state -> grid_state ;;
let tgrid_cont = make_ground "grid_cont";;

let mark_current_location s =
	if s.pendown then s.board.(s.x).(s.y) <- true;;

let move_forward_no_mark s =
	s.x <- max (min (s.x + s.dir.delta_x) (s.w-1)) 0;
	s.y <- max (min (s.y + s.dir.delta_y) (s.h-1)) 0;;

let move_forward s =
	move_forward_no_mark s;
	mark_current_location s;;

(* HACK seems influential on results to have these be constants vs variables
probably a performance issue that interacts with search timeouts? *)
let rotate_left = function
	|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
	|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
	|{delta_x = 0; delta_y = -1} -> {delta_x = 1; delta_y = 0}
	|{delta_x = 0; delta_y = 1} -> {delta_x = -1; delta_y = 0}
	|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled");;

let rotate_right = function
	|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
	|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
	|{delta_x = 0; delta_y = -1} -> {delta_x = -1; delta_y = 0}
	|{delta_x = 0; delta_y = 1} -> {delta_x = 1; delta_y = 0}
	|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled");;

let step_cost s =
	let cost = if (not s.pendown && not s.cost_when_penup) then 0. else 1. in
	s.reward <- s.reward -. cost;;

let ensure_location s =
		if s.x = -1 || s.y = -1 then raise (Exception "Location is not set.");;

ignore(primitive "grid_left" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		step_cost(s);
		s.dir <- (rotate_left s.dir);
		k(s)));;
ignore(primitive "grid_right" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		step_cost(s);
		s.dir <- (rotate_right s.dir);
		k(s)));;
ignore(primitive "grid_move" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		step_cost(s);
		move_forward(s);
		k(s)));;
ignore(primitive "grid_move_no_mark" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		step_cost(s);
		move_forward_no_mark(s);
		k(s)));;
ignore(primitive "grid_mark_current_location" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		step_cost(s);
		mark_current_location(s);
		k(s)));;
ignore(primitive "grid_dopendown" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		if s.cost_pen_change then step_cost(s);
		s.pendown <- true;
		k(s)));;
ignore(primitive "grid_dopenup" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);
		if s.cost_pen_change then step_cost(s);
		s.pendown <- false;
		k(s)));;
ignore(primitive "grid_setlocation" (tint @> tint @> tgrid_cont @> tgrid_cont)
	(fun (x: int) (y: int) (k: grid_cont) (s: grid_state) : grid_state ->
		if (
			s.x <> -1 || s.y <> -1 ||
			x < 0 || s.w <= x ||
			y < 0 || s.h <= y
		) then raise (Exception "TODO not valid") else
		s.x <- x;
		s.y <- y;
		mark_current_location s;
		k(s)));;

let print_row my_array=
	Printf.eprintf "[|";
	for i = 0 to ((Array.length my_array)-1) do
	   Printf.eprintf "%b" my_array.(i);
	done;
	Printf.eprintf "|]";;
let print_matrix the_matrix =
	Printf.eprintf "[|\n";
	for i = 0 to ((Array.length the_matrix)-1) do
		if not (phys_equal i 0) then Printf.eprintf "\n" else ();
		print_row the_matrix.(i);
	done;
	Printf.eprintf "|]\n";;

ignore(primitive "grid_embed" ((tgrid_cont @> tgrid_cont) @> tgrid_cont @> tgrid_cont)
	(fun (body: grid_cont -> grid_cont) (k: grid_cont) (s: grid_state) : grid_state ->
		ensure_location(s);

		(* save agent's state (location, orientation, pen) *)
		let x = s.x in
		let y = s.y in
		let pendown = s.pendown in
		let dir = s.dir in

		(* run the body *)
		let _ = body (fun s -> s) s in

		(* and once we've executed our body, restore agent state! *)
		s.x <- x;
		s.y <- y;
		s.pendown <- pendown;
		s.dir <- dir;

		(* also step cost? step_cost(s); *)

		(* execute rest of program *)
		let ns = k(s) in
		ns));;

ignore(primitive "grid_with_penup" ((tgrid_cont @> tgrid_cont) @> tgrid_cont @> tgrid_cont)
	(fun (body: grid_cont -> grid_cont) (k: grid_cont) (s: grid_state) : grid_state ->
		(* penup *)
		s.pendown <- false;
		(* run the body *)
		let _ = body (fun s -> s) s in
		(* pendown *)
		s.pendown <- true;
		(* execute rest of program *)
		k(s)));;

let evaluate_GRID timeout p state =
    begin
      (* Printf.eprintf "%s\n" (string_of_program p); *)
      let p = analyze_lazy_evaluation p in
      let new_discrete =
        try
          match run_for_interval
                  timeout
                  (fun () -> run_lazy_analyzed_with_arguments p [fun s -> s] state)
          with
          | Some(p) ->
            Some(p)
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

let score_binary final goal =
	let hit = (final.board = goal) in
	if hit then 0. else log 0.;;

let score_shortest_path final goal =
	let hit = (final.board = goal) in
	if hit then final.reward else log 0.;;

let score_progress_old partial_progress_weight invtemp final goal =
	let hit = (final.board = goal) in
	let sum_ = (fun (a : int array) : int -> (Array.fold_right ~f:(fun acc x -> acc + x) ~init:0 a)) in
	let map2rew = (fun (a : bool array) (b : bool array) : (int array) ->
		Array.mapi ~f:(fun idx x -> if x = b.(idx) then 0 else -1) a) in
	let (_, match_reward) = Array.fold_right ~f:(fun _ (idx, acc) ->
		(idx + 1, acc + (sum_ (map2rew goal.(idx) final.board.(idx))))
	) goal ~init:(0, 0) in
	(*Printf.eprintf "%s hit=%s rew=%f\n" (string_of_program p) (if hit then "true" else "false") (final.reward +. (if hit then 0. else -1000.));*)
	partial_progress_weight *. (float_of_int match_reward) +. invtemp *. final.reward +. (if hit then 0. else -1000.);;

let score_progress partial_progress_weight invtemp final goal =
	let hit = (final.board = goal) in
	let incorrect = ref 0 in
	let notdone = ref 0 in
	for x = 0 to ((Array.length goal)-1) do
		for y = 0 to ((Array.length goal.(0))-1) do
			if final.board.(x).(y) && not goal.(x).(y) then incorrect := !incorrect + 1;
			if not final.board.(x).(y) && goal.(x).(y) then notdone := !notdone + 1;
		done;
	done;
	if !incorrect <> 0 then log 0. else
	partial_progress_weight *. (-.(float_of_int !notdone)) +. invtemp *. final.reward +. (if hit then 0. else -1000.);;

register_special_task "GridTask" (fun extra ?timeout:(timeout = 0.001)
    name task_type examples ->
  assert (task_type = tgrid_cont @> tgrid_cont);
  assert (examples = []);

  let open Yojson.Basic.Util in
  let start : (bool array) array = extra |> member "start" |> to_list |>
		List.map ~f:(fun el -> el |> to_list |> List.map ~f:(fun el -> el |> to_bool) |> Array.of_list) |> Array.of_list
  in
  let goal : (bool array) array = extra |> member "goal" |> to_list |>
		List.map ~f:(fun el -> el |> to_list |> List.map ~f:(fun el -> el |> to_bool) |> Array.of_list) |> Array.of_list
	in
	let x = extra |> member "location" |> index 0 |> to_int
	in
	let y = extra |> member "location" |> index 1 |> to_int
	in
	let invtemp = extra |> member "invtemp" |> to_float
	in
	let try_all_start = extra |> member "try_all_start" |> to_bool in
	let partial_progress_weight = extra |> member "partial_progress_weight" |> to_float in
	let cost_pen_change = extra |> member "settings" |> member "cost_pen_change" |> to_bool in
	let cost_when_penup = extra |> member "settings" |> member "cost_when_penup" |> to_bool in
	let log_program = extra |> member "log_program" |> to_bool in

	let copyarr a = Array.map ~f:(Array.copy) a in
	let board_state start x y = {
		reward=0.; board=start; w=(Array.length start); h=(Array.length start.(0)); dir=up; pendown=true; x=x; y=y;
		cost_pen_change=cost_pen_change; cost_when_penup=cost_when_penup;
	} in

	let score_program p s x y =
		match (board_state s x y |> evaluate_GRID timeout p) with
			| Some(final) ->
				if partial_progress_weight <> 0. then (score_progress partial_progress_weight invtemp final goal) else
				if invtemp = 0. then (score_binary final goal) else (invtemp *. score_shortest_path final goal)
			(* if we can't execute, then we shouldn't consider this one *)
			| _ -> log 0.
	in

	let score_program_one_start p x y =
		(* copying here since we mutate in evaluation *)
		let s = copyarr start in
		score_program p s x y
	in

	let score_program_all_start p =
		let v = ref (log 0.) in
		for x = 0 to ((Array.length start)-1) do
			for y = 0 to ((Array.length start.(0))-1) do
				(* copying here since we mutate in evaluation *)
				let s = copyarr start in
				s.(x).(y) <- true;
				v := max !v (score_program p s x y);
			done;
		done;
		!v
	in

  (* Printf.eprintf "TARGETING:\n%s\n\n" *)

  { name = name    ;
    task_type = task_type ;
    log_likelihood = (fun p : float ->
			if log_program then Printf.eprintf "%s\n" (string_of_program p);
			if try_all_start then score_program_all_start p
			else score_program_one_start p x y)
  })
;;
