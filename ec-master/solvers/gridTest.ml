open Core
open Array

open Client
open Timeout
open Task
open Utils
open Program
open Type
open Grid

let () =
	let defaultTimeout = 0.1 in

	(*
	The order of these yojson imports is important; in particular, the to_string util used below is defined in both places.
	The one is Yojson.Basic will add quotes around the input? But we don't want that, we want to parse the JSON to a string
	*)
	let open Yojson.Basic in
	let open Yojson.Basic.Util in

	let rec unpack x =
		try magical (x |> to_int) with _ ->
		try magical (x |> to_number) with _ ->
		try magical (x |> to_bool) with _ ->
		try
			let v = x |> to_string in
			if String.length v = 1 then magical v.[0] else magical v
		with _ ->
		try
			x |> to_list |> List.map ~f:unpack |> magical
		with _ -> raise (Failure "could not unpack")
	in

	(* Compression.ml also has code for loading from file in argv *)
	let j = Yojson.Basic.from_channel Pervasives.stdin in

	(* First, we load the task *)
	let t = j |> member "task" in
	let e = t |> member "examples" |> to_list in
	let task_type = t |> member "request" |> deserialize_type in
	let examples = e |> List.map ~f:(fun ex -> (
		ex |> member "inputs" |> to_list |> List.map ~f:unpack,
		ex |> member "output" |> unpack)) in
	let name = t |> member "name" |> to_string in
	let special = t |> member "specialTask" |> to_string in
	let handler = special |> Hashtbl.find task_handler |> get_some in
	let task = handler (t |> member "extras") ~timeout:defaultTimeout name task_type examples in

	(* Then we load the program *)
	let program = j |> member "program" |> to_string |> parse_program |> get_some in

	(* And evaluate the program on the task! *)
	let ll = task.log_likelihood program in

	(* Generating output here *)
	let j = `Assoc(["logLikelihood",`Float(ll);]) in
	pretty_to_string j |> print_string;;
